#include "cpu_ops.h"

#include "cpu_bfloat16.h"
#include "cpu_fast_math.h"
#include "cpu_float8.h"
#include "cpu_mxfp4.h"
#include "cpu_qnk.h"
#include "cpu_vector.h"
#include "backends/ncnn/ncnn_linear.h"
#include "engine/cpu_thread_budget.h"
#include "engine/cpu_topology.h"
#include "ncnn/moe/runtime_config.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

bool simd_rms_norm_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags, RuntimeOptimizationCpuSimdRmsNorm);
}

bool cpu_fast_silu_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags, RuntimeOptimizationCpuFastSilu);
}

static bool cpu_mxfp4_bulk_row_pairs_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags, RuntimeOptimizationCpuMxfp4RowPairs);
}

static bool cpu_mxfp4_q8_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(
               optimization_flags,
               RuntimeOptimizationCpuMxfp4Q8)
           && mxfp4_q8_packed_kernel_available();
}

static bool cpu_packed_weights_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(
        optimization_flags,
        RuntimeOptimizationCpuPackedWeights);
}

static bool dense_host_storage_available(const TensorData& tensor) noexcept
{
    if (tensor.dtype == DType::Float32)
        return tensor.float32_values().size() == tensor.element_count();
    if (tensor.dtype == DType::BFloat16)
        return tensor.bfloat16_values().size() == tensor.element_count();
    if (tensor.dtype == DType::Float8E4M3)
        return tensor.float8_values().size() == tensor.element_count()
               && !tensor.quantization_scales.empty();
    return true;
}

static void require_dense_host_storage(
    const TensorData& tensor,
    const char* role)
{
    if (!dense_host_storage_available(tensor))
    {
        throw std::runtime_error(
            std::string("Vulkan ") + role
            + " execution failed after its host storage was released");
    }
}

float bfloat16_to_float(uint16_t value) noexcept
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t float_to_bfloat16(float value) noexcept
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

static float tensor_value(const TensorData& tensor, size_t index)
{
    if (tensor.dtype == DType::Float32)
        return tensor.float32_values()[index];
    if (tensor.dtype == DType::BFloat16)
        return bfloat16_to_float(tensor.bfloat16_values()[index]);
    if (tensor.dtype == DType::Float8E4M3)
        return float8_e4m3_to_float(tensor.float8_values()[index]);
    assert(false && "tensor_value only supports float32 and bfloat16");
    return 0.0f;
}

static float exact_scaled_silu(float value, float sigmoid_scale) noexcept
{
    return value / (1.0f + float_approximate_exp(-sigmoid_scale * value));
}

// Degree-7 approximation over one range-reduced log2 interval.
float approximate_scaled_silu(float value, float sigmoid_scale) noexcept
{
    const float scaled_value = sigmoid_scale * value;
    if (scaled_value >= 0.0f)
        return value / (1.0f + float_approximate_exp(-scaled_value));
    const float exponential = float_approximate_exp(scaled_value);
    return value * exponential / (1.0f + exponential);
}

static float selected_scaled_silu(float value, float sigmoid_scale, bool approximate) noexcept
{
    return approximate ? approximate_scaled_silu(value, sigmoid_scale) : exact_scaled_silu(value, sigmoid_scale);
}

float scaled_silu(
    float value,
    float sigmoid_scale,
    uint64_t optimization_flags) noexcept
{
    return selected_scaled_silu(
        value,
        sigmoid_scale,
        cpu_fast_silu_enabled(optimization_flags));
}

const char* scaled_silu_kernel_name(uint64_t optimization_flags) noexcept
{
    return cpu_fast_silu_enabled(optimization_flags)
               ? "polynomial"
               : "ncnn-fast-exp";
}

static bool allow_openmp_parallel_region() noexcept
{
#if defined(_OPENMP)
    return omp_in_parallel() == 0;
#else
    return false;
#endif
}

static uint32_t physical_core_count()
{
    static const uint32_t count = discover_cpu_topology().physical_core_count;
    return count;
}

static uint32_t select_cpu_linear_thread_limit() noexcept
{
#if defined(_OPENMP)
    const uint32_t maximum_threads = cpu_openmp_thread_limit();
    const uint32_t core_count = physical_core_count();
    return core_count == 0 ? maximum_threads : std::min(maximum_threads, core_count);
#else
    return 1;
#endif
}

uint32_t cpu_linear_thread_limit() noexcept
{
    // Honor the scheduler's thread-local OpenMP cap.
    return select_cpu_linear_thread_limit();
}

uint32_t float8_linear_thread_limit() noexcept
{
    return cpu_linear_thread_limit();
}

static int openmp_linear_team_size(uint64_t operation_count, DType dtype) noexcept
{
#if defined(_OPENMP)
    // Scale the OpenMP team by operation count.
    static constexpr uint64_t minimum_parallel_operations = 1024 * 1024;
    static constexpr uint64_t minimum_dense_parallel_operations = 2 * 1024 * 1024;
    static constexpr uint64_t full_team_operations = 8 * 1024 * 1024;
    const uint64_t minimum_operations = dtype == DType::Float8E4M3 ? minimum_parallel_operations : minimum_dense_parallel_operations;
    if (omp_in_parallel() != 0 || operation_count < minimum_operations)
    {
        return 1;
    }
    const int maximum_threads = static_cast<int>(dtype == DType::Float8E4M3 ? float8_linear_thread_limit() : cpu_linear_thread_limit());
    if (operation_count < full_team_operations)
        return std::min(4, maximum_threads);
    return maximum_threads;
#else
    (void)operation_count;
    (void)dtype;
    return 1;
#endif
}

static int openmp_mxfp4_group_team_size(uint64_t operation_count) noexcept
{
#if defined(_OPENMP)
    static constexpr uint64_t minimum_operations_per_thread = 128 * 1024;
    const uint32_t core_count = physical_core_count();
    const int maximum_threads = static_cast<int>(cpu_linear_thread_limit());
    const uint64_t useful_threads = std::max<uint64_t>(1, (operation_count + minimum_operations_per_thread - 1) / minimum_operations_per_thread);
    const int topology_limit = core_count == 0 ? maximum_threads : std::min(maximum_threads, static_cast<int>(core_count));
    return std::max(1, std::min(topology_limit, static_cast<int>(std::min<uint64_t>(useful_threads, static_cast<uint64_t>(std::numeric_limits<int>::max())))));
#else
    (void)operation_count;
    return 1;
#endif
}

static float expert_activation(
    float value,
    ExpertActivation activation,
    float limit,
    uint64_t optimization_flags) noexcept
{
    switch (activation)
    {
    case ExpertActivation::Relu: return std::max(0.0f, value);
    case ExpertActivation::Silu:
        return scaled_silu(value, 1.0f, optimization_flags);
    case ExpertActivation::Gelu:
        return 0.5f * value
               * (1.0f + std::erf(value / std::sqrt(2.0f)));
    case ExpertActivation::ClampedSilu:
    {
        const float clamped = limit > 0.0f ? std::clamp(value, -limit, limit) : value;
        return scaled_silu(clamped, 1.0f, optimization_flags);
    }
    case ExpertActivation::DeepSeekSwiGlu:
    {
        const float clamped = limit > 0.0f ? std::min(value, limit) : value;
        return scaled_silu(clamped, 1.0f, optimization_flags);
    }
    case ExpertActivation::GptOssSwiGlu: return value;
    }
    return value;
}

static void apply_float8_gate_up_activation(
    float* output,
    const float* gate,
    const float* up,
    uint32_t count,
    ExpertActivation activation,
    float activation_limit,
    uint64_t optimization_flags)
{
    if (cpu_fast_silu_enabled(optimization_flags)
        && activation_limit <= 0.0f
        && (activation == ExpertActivation::Silu
            || activation == ExpertActivation::DeepSeekSwiGlu))
    {
        float_silu_mul(output, gate, up, 1.0f, 0.0f, count);
        return;
    }

    for (uint32_t column = 0; column < count; ++column)
    {
        float up_value = up[column];
        if (activation == ExpertActivation::DeepSeekSwiGlu
            && activation_limit > 0.0f)
        {
            up_value = std::clamp(up_value, -activation_limit,
                                  activation_limit);
        }
        output[column] = up_value * expert_activation(gate[column], activation, activation_limit, optimization_flags);
    }
}

void embedding_batch_into(const TensorData& embedding, std::span<const int32_t> input_ids, CpuBatch& output)
{
    assert(embedding.shape.size() == 2);
    const uint32_t hidden_size = embedding.shape[1];
    output.reset(input_ids.size(), hidden_size, false);
    for (size_t token_index = 0; token_index < input_ids.size(); ++token_index)
    {
        const size_t offset = static_cast<size_t>(input_ids[token_index]) * hidden_size;
        float* destination = output.row(token_index);
        for (uint32_t column = 0; column < hidden_size; ++column)
            destination[column] = tensor_value(embedding, offset + column);
    }
}

static void prepare_float8_input(CpuBatch& scratch, const CpuBatch& input)
{
    // Quantized activation scratch belongs to the operation caller.
    scratch.reset(input.rows(), input.columns(), false);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        std::copy_n(input.row(row), input.columns(), scratch.row(row));
    }
}

static void prepare_quantized_float8_input(CpuBatch& scratch, const CpuBatch& input, uint64_t optimization_flags)
{
    scratch.reset(input.rows(), input.columns(), false);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        quantize_float8_e4m3(
            input.row(row), scratch.row(row), input.columns(), 128, true, optimization_flags);
    }
}

static bool cpu_float8_fused_projections_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags,
                                        RuntimeOptimizationCpuFloat8FusedProjections);
}

static void float8_linear_group_into(
    const TensorData& matrix,
    const CpuBatch& quantized_input,
    CpuBatch& output,
    uint32_t first_output_column,
    uint32_t group_size,
    uint32_t input_blocks,
    uint32_t block_size,
    uint64_t optimization_flags)
{
    const uint32_t input_columns = matrix.shape[1];
    const std::span<const uint8_t> matrix_values = matrix.float8_values();
    const uint8_t* weights = matrix_values.data()
                             + static_cast<size_t>(first_output_column) * input_columns;
    const float* scales = matrix.quantization_scales.data()
                          + static_cast<size_t>(first_output_column / block_size) * input_blocks;
    if (quantized_input.rows() > 1)
    {
        float8_e4m3_quantized_input_dot_rows_batch(
            weights, input_columns, scales, quantized_input.row(0),
            quantized_input.columns(), input_columns, block_size, group_size,
            output.columns(), quantized_input.rows(),
            output.row(0) + first_output_column, optimization_flags);
        return;
    }
    for (size_t token_index = 0; token_index < quantized_input.rows();
         ++token_index)
    {
        if (group_size == 1)
        {
            output.row(token_index)[first_output_column] = float8_e4m3_quantized_input_dot(
                weights, scales, quantized_input.row(token_index),
                input_columns, block_size, optimization_flags);
        }
        else
        {
            float8_e4m3_quantized_input_dot_rows(
                weights, input_columns, scales,
                quantized_input.row(token_index), input_columns, block_size,
                group_size, output.row(token_index) + first_output_column,
                optimization_flags);
        }
    }
}

static uint32_t float8_linear_row_group_size_for_shape(
    const TensorData& matrix,
    size_t token_count,
    uint64_t optimization_flags) noexcept;

static void float8_linear_quantized_into(
    const TensorData& matrix,
    const CpuBatch& quantized_input,
    CpuBatch& output,
    uint64_t optimization_flags)
{
    assert(matrix.shape.size() == 2);
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    assert(quantized_input.columns() == input_columns);
    constexpr uint32_t block_size = 128;
    const uint32_t input_blocks = (input_columns + block_size - 1) / block_size;
    const uint32_t output_blocks = (output_columns + block_size - 1) / block_size;
    const std::span<const uint8_t> matrix_values = matrix.float8_values();
    assert(matrix_values.size() == matrix.element_count());
    assert(matrix.quantization_scales.size()
           == static_cast<size_t>(output_blocks) * input_blocks);
    (void)output_blocks;

    output.reset(quantized_input.rows(), output_columns, false);
    const uint64_t operation_count = static_cast<uint64_t>(output_columns) * input_columns
                                     * quantized_input.rows();
    const int linear_team_size = openmp_linear_team_size(operation_count, matrix.dtype);
    const bool parallelize_linear = linear_team_size > 1;
    const int64_t parallel_output_columns = static_cast<int64_t>(output_columns);

    const uint32_t row_group_size = float8_linear_row_group_size_for_shape(
        matrix, quantized_input.rows(), optimization_flags);
#pragma omp parallel for num_threads(linear_team_size) if (parallelize_linear)
    for (int64_t output_group = 0;
         output_group < (parallel_output_columns + row_group_size - 1)
                            / row_group_size;
         ++output_group)
    {
        const uint32_t first_output_column = static_cast<uint32_t>(output_group) * row_group_size;
        const uint32_t group_size = std::min(row_group_size,
                                             output_columns - first_output_column);
        float8_linear_group_into(
            matrix, quantized_input, output, first_output_column, group_size,
            input_blocks, block_size, optimization_flags);
    }
}

static bool float32_linear_gemm_tile_into(
    const TensorData& matrix,
    const CpuBatch& input,
    CpuBatch& output,
    int team_size)
{
    // GEMV stays row-parallel; larger batches use an MxN tile.
    if (input.rows() < 2 || matrix.shape[0] < 2 || matrix.shape[1] < 16)
        return false;
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t output_tile = output_columns >= 8 ? 8 : 4;
    const uint32_t output_groups = (output_columns + output_tile - 1) / output_tile;
    const uint32_t token_groups = static_cast<uint32_t>((input.rows() + 3) / 4);
    const int64_t tile_count = static_cast<int64_t>(output_groups) * token_groups;
    const std::span<const float> weights = matrix.float32_values();
#pragma omp parallel for num_threads(team_size) if (team_size > 1)
    for (int64_t tile = 0; tile < tile_count; ++tile)
    {
        const uint32_t output_group = static_cast<uint32_t>(tile % output_groups);
        const uint32_t token_group = static_cast<uint32_t>(tile / output_groups);
        const uint32_t first_output = output_group * output_tile;
        const uint32_t first_token = token_group * 4;
        const uint32_t valid_outputs = std::min(output_tile, output_columns - first_output);
        const uint32_t valid_tokens = std::min<uint32_t>(4, static_cast<uint32_t>(input.rows()) - first_token);
        if (output_tile == 8)
        {
            float_gemm_4x8(
                weights.data() + static_cast<size_t>(first_output) * input_columns,
                input_columns,
                input.row(first_token),
                input.columns(),
                input_columns,
                valid_outputs,
                valid_tokens,
                output.row(first_token) + first_output,
                output.columns());
        }
        else
        {
            float_gemm_4x4(
                weights.data() + static_cast<size_t>(first_output) * input_columns,
                input_columns,
                input.row(first_token),
                input.columns(),
                input_columns,
                valid_outputs,
                valid_tokens,
                output.row(first_token) + first_output,
                output.columns());
        }
    }
    return true;
}

static bool bfloat16_linear_gemm_tile_into(
    const TensorData& matrix,
    const CpuBatch& input,
    CpuBatch& output,
    int team_size)
{
    if (input.rows() < 2 || matrix.shape[0] < 2 || matrix.shape[1] < 16)
        return false;
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t output_tile = std::min(8u, output_columns);
    const uint32_t output_groups = (output_columns + output_tile - 1) / output_tile;
    const uint32_t token_groups = static_cast<uint32_t>((input.rows() + 3) / 4);
    const int64_t tile_count = static_cast<int64_t>(output_groups) * token_groups;
    const std::span<const uint16_t> weights = matrix.bfloat16_values();
#pragma omp parallel for num_threads(team_size) if (team_size > 1)
    for (int64_t tile = 0; tile < tile_count; ++tile)
    {
        const uint32_t output_group = static_cast<uint32_t>(tile % output_groups);
        const uint32_t token_group = static_cast<uint32_t>(tile / output_groups);
        const uint32_t first_output = output_group * output_tile;
        const uint32_t first_token = token_group * 4;
        const uint32_t valid_outputs = std::min(output_tile, output_columns - first_output);
        const uint32_t valid_tokens = std::min<uint32_t>(4, static_cast<uint32_t>(input.rows()) - first_token);
        bfloat16_gemm_4x8(
            weights.data() + static_cast<size_t>(first_output) * input_columns,
            input_columns,
            input.row(first_token),
            input.columns(),
            input_columns,
            valid_outputs,
            valid_tokens,
            output.row(first_token) + first_output,
            output.columns());
    }
    return true;
}

static uint32_t float8_linear_row_group_size_for_shape(
    const TensorData& matrix,
    size_t token_count,
    uint64_t optimization_flags) noexcept
{
    const uint32_t kernel_group = float8_linear_row_group_size(optimization_flags);
    if (token_count == 1 || matrix.shape[0] < 8)
        return kernel_group;
    // Use a wider output tile when activation reuse is available.
    return std::max(8u, kernel_group);
}

bool float8_linear_pair_batch_into(
    const TensorData& first,
    const TensorData& second,
    const CpuBatch& input,
    CpuBatch& first_output,
    CpuBatch& second_output,
    uint64_t optimization_flags,
    const CompiledOperator* first_executable,
    const CompiledOperator* second_executable)
{
    if (!cpu_float8_fused_projections_enabled(optimization_flags)
        || first.dtype != DType::Float8E4M3
        || second.dtype != DType::Float8E4M3
        || first.shape.size() != 2 || second.shape.size() != 2
        || first.shape[1] != input.columns()
        || second.shape[1] != input.columns()
        || (first_executable && (first_executable->float8 || first_executable->linear))
        || (second_executable && (second_executable->float8 || second_executable->linear))
        || !dense_host_storage_available(first)
        || !dense_host_storage_available(second))
    {
        return false;
    }

    // Keep the paired path free of decoded-weight side storage.
    constexpr uint32_t block_size = 128;
    const uint32_t first_input_blocks = (first.shape[1] + block_size - 1) / block_size;
    const uint32_t second_input_blocks = (second.shape[1] + block_size - 1) / block_size;
    const uint32_t first_output_blocks = (first.shape[0] + block_size - 1) / block_size;
    const uint32_t second_output_blocks = (second.shape[0] + block_size - 1) / block_size;
    if (first.float8_values().size() != first.element_count()
        || second.float8_values().size() != second.element_count()
        || first.quantization_scales.size()
               != static_cast<size_t>(first_output_blocks)
                      * first_input_blocks
        || second.quantization_scales.size()
               != static_cast<size_t>(second_output_blocks)
                      * second_input_blocks)
    {
        return false;
    }

    first_output.reset(input.rows(), first.shape[0], false);
    second_output.reset(input.rows(), second.shape[0], false);
    CpuBatch quantized_input;
    prepare_quantized_float8_input(quantized_input, input, optimization_flags);

    const uint32_t row_group_size = float8_linear_row_group_size(optimization_flags);
    const int64_t first_group_count = (static_cast<int64_t>(first.shape[0]) + row_group_size - 1)
                                      / row_group_size;
    const int64_t second_group_count = (static_cast<int64_t>(second.shape[0]) + row_group_size - 1)
                                       / row_group_size;
    uint64_t operation_count = static_cast<uint64_t>(first.shape[0]) * first.shape[1]
                               * input.rows();
    const uint64_t second_operations = static_cast<uint64_t>(second.shape[0]) * second.shape[1]
                                       * input.rows();
    if (operation_count > std::numeric_limits<uint64_t>::max()
                              - second_operations)
    {
        operation_count = std::numeric_limits<uint64_t>::max();
    }
    else
    {
        operation_count += second_operations;
    }
    const int team_size = openmp_linear_team_size(operation_count, DType::Float8E4M3);
    const bool parallelize = team_size > 1;
#pragma omp parallel num_threads(team_size) if (parallelize)
    {
#pragma omp for schedule(static)
        for (int64_t output_group = 0; output_group < first_group_count;
             ++output_group)
        {
            const uint32_t first_output_column = static_cast<uint32_t>(output_group) * row_group_size;
            const uint32_t group_size = std::min(row_group_size,
                                                 first.shape[0] - first_output_column);
            float8_linear_group_into(
                first, quantized_input, first_output, first_output_column,
                group_size, first_input_blocks, block_size, optimization_flags);
        }
#pragma omp for schedule(static)
        for (int64_t output_group = 0; output_group < second_group_count;
             ++output_group)
        {
            const uint32_t first_output_column = static_cast<uint32_t>(output_group) * row_group_size;
            const uint32_t group_size = std::min(row_group_size,
                                                 second.shape[0] - first_output_column);
            float8_linear_group_into(
                second, quantized_input, second_output, first_output_column,
                group_size, second_input_blocks, block_size, optimization_flags);
        }
    }
    return true;
}

bool float8_linear_rms_norm_batch_into(
    const TensorData& matrix,
    const CpuBatch& input,
    const TensorData& norm_weight,
    float epsilon,
    CpuBatch& output,
    uint64_t optimization_flags,
    const CompiledOperator* executable)
{
    if (!cpu_float8_fused_projections_enabled(optimization_flags)
        || matrix.dtype != DType::Float8E4M3 || matrix.shape.size() != 2
        || matrix.shape[1] != input.columns()
        || (executable && (executable->float8 || executable->linear))
        || (norm_weight.dtype != DType::Float32
            && norm_weight.dtype != DType::BFloat16)
        || norm_weight.element_count() != input.columns()
        || !dense_host_storage_available(matrix)
        || !dense_host_storage_available(norm_weight))
    {
        return false;
    }

    CpuBatch quantized_input;
    prepare_float8_input(quantized_input, input);
    const bool use_simd = simd_rms_norm_enabled(optimization_flags);
    for (size_t token_index = 0; token_index < input.rows(); ++token_index)
    {
        const float* source = input.row(token_index);
        float* destination = quantized_input.row(token_index);
        if (use_simd && norm_weight.dtype == DType::Float32)
        {
            float_rms_norm(
                destination,
                source,
                norm_weight.float32_values().data(),
                epsilon,
                0.0f,
                input.columns());
        }
        else if (use_simd && norm_weight.dtype == DType::BFloat16)
        {
            bfloat16_rms_norm(
                destination,
                source,
                norm_weight.bfloat16_values().data(),
                epsilon,
                0.0f,
                input.columns());
        }
        else
        {
            const float square_sum = std::inner_product(
                source,
                source + input.columns(), source,
                0.0f);
            const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(input.columns()) + epsilon);
            for (uint32_t column = 0; column < input.columns(); ++column)
            {
                const float weight_value = norm_weight.dtype == DType::Float32
                                                ? norm_weight.float32_values()[column]
                                                : bfloat16_to_float(norm_weight.bfloat16_values()[column]);
                destination[column] = source[column] * inverse_rms * weight_value;
            }
        }
        quantize_float8_e4m3_inplace(
            destination, input.columns(), 128, true, optimization_flags);
    }

    float8_linear_quantized_into(matrix, quantized_input, output, optimization_flags);
    return true;
}

static std::shared_ptr<const Mxfp4Q8PackedMatrix> get_mxfp4_q8_packed_weights(
    const TensorData& matrix,
    uint32_t block_count,
    size_t row_count)
{
    // Protect the first immutable sidecar build.
    static std::mutex build_locks[64];
    const uintptr_t storage_key = reinterpret_cast<uintptr_t>(
        matrix.mxfp4_blocks.data());
    std::lock_guard<std::mutex> build_lock(
        build_locks[(storage_key >> 6) & 63u]);
    std::shared_ptr<const Mxfp4Q8PackedMatrix> cached = matrix.mxfp4_q8_packed;
    if (cached && cached->valid()
        && cached->rows == row_count
        && cached->block_count == block_count)
    {
        return cached;
    }

    auto packed = std::make_shared<Mxfp4Q8PackedMatrix>();
    if (!mxfp4_q8_pack_weights(
            matrix.mxfp4_blocks.data(),
            matrix.mxfp4_scales.data(),
            block_count,
            row_count,
            *packed))
    {
        return {};
    }
    std::shared_ptr<const Mxfp4Q8PackedMatrix> desired = packed;
    matrix.mxfp4_q8_packed = desired;
    return desired;
}

void linear_batch_into(const TensorData& matrix, const CpuBatch& input, CpuBatch& output, uint64_t optimization_flags, const CompiledOperator* executable, ExecutionBackend backend)
{
    assert(matrix.shape.size() == 2);
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    assert(input.columns() == input_columns);

    if (is_qnk_dtype(matrix.dtype))
    {
        if (backend == ExecutionBackend::Vulkan
            && executable
            && executable->qnk
            && executable->qnk->forward(input, output))
        {
            return;
        }
        if (qnk_linear_batch_into(
                matrix,
                input,
                output,
                cpu_packed_weights_enabled(optimization_flags)))
            return;
    }

    if (backend == ExecutionBackend::Vulkan
        && ((executable && executable->bfloat16
             && executable->bfloat16->forward(input, output))
            || (executable && executable->float8 && executable->float8->forward(input, output))
            || (executable && executable->qnk && executable->qnk->forward(input, output))
            || (executable && executable->linear && executable->linear->forward(input, output))))
        return;
    require_dense_host_storage(matrix, "linear weight");
    output.reset(input.rows(), output_columns, false);
    const int64_t parallel_output_columns = static_cast<int64_t>(output_columns);
    const uint64_t operation_count = static_cast<uint64_t>(output_columns) * input_columns * input.rows();
    const int linear_team_size = openmp_linear_team_size(operation_count, matrix.dtype);
    const bool parallelize_linear = linear_team_size > 1;
    if (matrix.dtype == DType::BFloat16
        && bfloat16_batched_linear(matrix.bfloat16_values().data(),
                                   input.row(0),
                                   input.columns(),
                                   input.rows(),
                                   output_columns,
                                   input_columns,
                                   output.row(0),
                                   output.columns(),
                                   linear_team_size,
                                   optimization_flags))
    {
        return;
    }
    if (matrix.dtype == DType::Float32
        && float32_linear_gemm_tile_into(matrix, input, output, linear_team_size))
    {
        return;
    }
    if (matrix.dtype == DType::BFloat16
        && bfloat16_linear_gemm_tile_into(matrix, input, output, linear_team_size))
    {
        return;
    }
    if (matrix.dtype == DType::Float32)
    {
        const std::span<const float> matrix_values = matrix.float32_values();
#pragma omp parallel for num_threads(linear_team_size) if (parallelize_linear)
        for (int64_t output_column = 0; output_column < parallel_output_columns; ++output_column)
        {
            const float* weights = matrix_values.data() + static_cast<size_t>(output_column) * input_columns;
            for (size_t token_index = 0; token_index < input.rows(); ++token_index)
            {
                const float* token = input.row(token_index);
                output.row(token_index)[output_column] = float_dot(weights, token, input_columns);
            }
        }
    }
    else if (matrix.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> matrix_values = matrix.bfloat16_values();
#pragma omp parallel for num_threads(linear_team_size) if (parallelize_linear)
        for (int64_t output_column = 0;
             output_column < parallel_output_columns; ++output_column)
        {
            const uint16_t* weights = matrix_values.data()
                                      + static_cast<size_t>(output_column) * input_columns;
            for (size_t token_index = 0; token_index < input.rows(); ++token_index)
            {
                const float* token = input.row(token_index);
                output.row(token_index)[output_column] = bfloat16_dot(weights, token, input_columns);
            }
        }
    }
    else if (matrix.dtype == DType::Float8E4M3)
    {
        CpuBatch quantized_input;
        prepare_quantized_float8_input(quantized_input, input, optimization_flags);
        float8_linear_quantized_into(matrix, quantized_input, output, optimization_flags);
    }
    else if (matrix.dtype == DType::Int8)
    {
        const std::span<const int8_t> matrix_values = matrix.int8_values();
#pragma omp parallel for num_threads(linear_team_size) if (parallelize_linear)
        for (int64_t output_column = 0; output_column < parallel_output_columns; ++output_column)
        {
            const int8_t* weights = matrix_values.data() + static_cast<size_t>(output_column) * input_columns;
            const float scale = matrix.quantization_scales[output_column];
            for (size_t token_index = 0; token_index < input.rows(); ++token_index)
            {
                const float* token = input.row(token_index);
                output.row(token_index)[output_column] = int8_float_dot(weights, token, input_columns) * scale;
            }
        }
    }
    else if (matrix.dtype == DType::MxFp4)
    {
        const uint32_t blocks_per_row = input_columns / 32;
        const int64_t row_pair_count = static_cast<int64_t>((output_columns + 1) / 2);
        Mxfp4Q8Batch q8_input;
        const bool use_q8 = cpu_mxfp4_q8_enabled(optimization_flags)
                            && input_columns % 32 == 0;
        if (use_q8)
        {
            mxfp4_q8_quantize_batch(
                input.row(0),
                input.columns(),
                input.rows(),
                input_columns,
                q8_input);

            // Repack complete matrices; small tails keep the row-pair path.
            if (cpu_packed_weights_enabled(optimization_flags)
                && output_columns >= 4
                && mxfp4_q8_packed_kernel_available())
            {
                const std::shared_ptr<const Mxfp4Q8PackedMatrix> packed_weights = get_mxfp4_q8_packed_weights(
                    matrix,
                    blocks_per_row,
                    output_columns);
                if (packed_weights)
                {
                    mxfp4_q8_packed_gemm(
                        *packed_weights,
                        q8_input.row(0),
                        q8_input.columns,
                        q8_input.row_scales(0),
                        (input_columns + 31) / 32,
                        input.rows(),
                        output.row(0),
                        output.columns());
                    return;
                }
            }
        }
#pragma omp parallel for num_threads(linear_team_size) if (parallelize_linear)
        for (int64_t row_pair = 0; row_pair < row_pair_count; ++row_pair)
        {
            const uint32_t first_row = static_cast<uint32_t>(row_pair) * 2;
            const uint32_t second_row = first_row + 1;
            const uint8_t* first_blocks = matrix.mxfp4_blocks.data() + static_cast<size_t>(first_row) * input_columns / 2;
            const uint8_t* first_scales = matrix.mxfp4_scales.data() + static_cast<size_t>(first_row) * blocks_per_row;
            if (second_row < output_columns)
            {
                const uint8_t* second_blocks = matrix.mxfp4_blocks.data() + static_cast<size_t>(second_row) * input_columns / 2;
                const uint8_t* second_scales = matrix.mxfp4_scales.data() + static_cast<size_t>(second_row) * blocks_per_row;
                if (use_q8)
                {
                    mxfp4_q8_matmul_rows2(
                        first_blocks,
                        first_scales,
                        second_blocks,
                        second_scales,
                        blocks_per_row,
                        q8_input.row(0),
                        input.columns(),
                        q8_input.row_scales(0),
                        (input_columns + 31) / 32,
                        input.rows(),
                        output.row(0) + first_row,
                        output.columns(),
                        output.row(0) + second_row,
                        output.columns());
                }
                else
                {
                    mxfp4_matmul_rows2(first_blocks, first_scales, second_blocks, second_scales, blocks_per_row, input.row(0), input.columns(), input.rows(),
                                       output.row(0) + first_row, output.columns(), output.row(0) + second_row, output.columns());
                }
            }
            else if (input.rows() == 1)
            {
                output.row(0)[first_row] = use_q8
                                               ? mxfp4_q8_dot(first_blocks, first_scales, blocks_per_row, q8_input.row(0), q8_input.row_scales(0))
                                               : mxfp4_dot(first_blocks, first_scales, blocks_per_row, input.row(0));
            }
            else
            {
                if (use_q8)
                {
                    mxfp4_q8_gemm_row(
                        first_blocks,
                        first_scales,
                        blocks_per_row,
                        q8_input.row(0),
                        input.columns(),
                        q8_input.row_scales(0),
                        (input_columns + 31) / 32,
                        input.rows(),
                        output.row(0) + first_row,
                        output.columns());
                }
                else
                {
                    mxfp4_gemm_row(first_blocks, first_scales, blocks_per_row, input.row(0), input.columns(), input.rows(), output.row(0) + first_row,
                                   output.columns());
                }
            }
        }
    }
    else
    {
        assert(false && "unsupported matrix dtype");
    }
}

CpuBatch linear_batch(const TensorData& matrix, const CpuBatch& input, uint64_t optimization_flags, const CompiledOperator* executable, ExecutionBackend backend)
{
    CpuBatch output;
    linear_batch_into(matrix, input, output, optimization_flags, executable, backend);
    return output;
}

bool fused_float8_gate_up_batch(const TensorData& gate,
                                const TensorData& up,
                                const CpuBatch& input,
                                ExpertActivation activation,
                                float activation_limit,
                                CpuBatch& output,
                                uint64_t optimization_flags,
                                const CompiledOperator* gate_executable,
                                const CompiledOperator* up_executable)
{
    if (gate.dtype != DType::Float8E4M3
        || up.dtype != DType::Float8E4M3
        || gate.shape.size() != 2
        || up.shape != gate.shape
        || gate.shape[1] != input.columns()
        || (gate_executable && (gate_executable->float8 || gate_executable->linear))
        || (up_executable && (up_executable->float8 || up_executable->linear))
        || !dense_host_storage_available(gate)
        || !dense_host_storage_available(up))
    {
        return false;
    }

    constexpr uint32_t block_size = 128;
    const uint32_t output_columns = gate.shape[0];
    const uint32_t input_columns = gate.shape[1];
    const uint32_t input_blocks = (input_columns + block_size - 1) / block_size;
    const uint32_t output_blocks = (output_columns + block_size - 1) / block_size;
    if (gate.float8_values().size() != gate.element_count()
        || up.float8_values().size() != up.element_count()
        || gate.quantization_scales.size()
               != static_cast<size_t>(output_blocks) * input_blocks
        || up.quantization_scales.size()
               != static_cast<size_t>(output_blocks) * input_blocks)
    {
        return false;
    }

    CpuBatch quantized_input;
    prepare_quantized_float8_input(quantized_input, input, optimization_flags);

    output.reset(input.rows(), output_columns, false);
    const uint32_t row_group_size = float8_linear_row_group_size(optimization_flags);
    const int64_t output_group_count = (static_cast<int64_t>(output_columns) + row_group_size - 1)
                                       / row_group_size;
    const uint64_t operation_count = static_cast<uint64_t>(output_columns) * input_columns * input.rows()
                                     * 2;
    const int team_size = openmp_linear_team_size(operation_count, DType::Float8E4M3);
    const bool parallelize = team_size > 1;
    const std::span<const uint8_t> gate_values = gate.float8_values();
    const std::span<const uint8_t> up_values = up.float8_values();
#pragma omp parallel for num_threads(team_size) if (parallelize)
    for (int64_t output_group = 0; output_group < output_group_count;
         ++output_group)
    {
        const uint32_t first_output_column = static_cast<uint32_t>(output_group) * row_group_size;
        const uint32_t group_size = std::min(row_group_size, output_columns - first_output_column);
        const uint8_t* gate_weights = gate_values.data()
                                      + static_cast<size_t>(first_output_column) * input_columns;
        const uint8_t* up_weights = up_values.data()
                                    + static_cast<size_t>(first_output_column) * input_columns;
        const float* gate_scales = gate.quantization_scales.data()
                                   + static_cast<size_t>(first_output_column / block_size)
                                         * input_blocks;
        const float* up_scales = up.quantization_scales.data()
                                 + static_cast<size_t>(first_output_column / block_size)
                                       * input_blocks;
        for (size_t token_index = 0; token_index < input.rows();
             ++token_index)
        {
            float gate_output[8] = {};
            float up_output[8] = {};
            if (row_group_size == 1)
            {
                gate_output[0] = float8_e4m3_quantized_input_dot(
                    gate_weights, gate_scales,
                    quantized_input.row(token_index), input_columns,
                    block_size, optimization_flags);
                up_output[0] = float8_e4m3_quantized_input_dot(
                    up_weights, up_scales,
                    quantized_input.row(token_index), input_columns,
                    block_size, optimization_flags);
            }
            else
            {
                float8_e4m3_quantized_input_dot_rows(
                    gate_weights, input_columns, gate_scales,
                    quantized_input.row(token_index), input_columns,
                    block_size, group_size, gate_output, optimization_flags);
                float8_e4m3_quantized_input_dot_rows(
                    up_weights, input_columns, up_scales,
                    quantized_input.row(token_index), input_columns,
                    block_size, group_size, up_output, optimization_flags);
            }
            float* destination = output.row(token_index)
                                 + first_output_column;
            apply_float8_gate_up_activation(
                destination, gate_output, up_output, group_size, activation,
                activation_limit, optimization_flags);
        }
    }
    return true;
}

CpuBatch fused_mxfp4_gate_up_batch(const TensorData& matrix, const TensorData* bias, const CpuBatch& input, ExpertActivation activation, float activation_limit,
                                   uint64_t optimization_flags)
{
    assert(matrix.dtype == DType::MxFp4 && matrix.shape.size() == 2);
    assert(matrix.shape[0] % 2 == 0 && matrix.shape[1] == input.columns());
    const bool approximate_activation = cpu_fast_silu_enabled(optimization_flags);
    const uint32_t intermediate_size = matrix.shape[0] / 2;
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t blocks_per_row = input_columns / 32;
    if (bias)
        assert(bias->shape == std::vector<uint32_t>{matrix.shape[0]});

    CpuBatch output;
    output.reset(input.rows(), intermediate_size, false);
    Mxfp4Q8Batch q8_input;
    const bool use_q8 = cpu_mxfp4_q8_enabled(optimization_flags)
                        && input_columns % 32 == 0;
    if (use_q8)
    {
        mxfp4_q8_quantize_batch(
            input.row(0),
            input.columns(),
            input.rows(),
            input_columns,
            q8_input);
    }
    const auto matmul_rows2 = [&](const uint8_t* first_packed,
                                  const uint8_t* first_scales,
                                  const uint8_t* second_packed,
                                  const uint8_t* second_scales,
                                  uint32_t block_count,
                                  const float* float_input,
                                  size_t input_stride,
                                  size_t token_count,
                                  float* first_output,
                                  size_t first_output_stride,
                                  float* second_output,
                                  size_t second_output_stride) {
        if (use_q8)
        {
            mxfp4_q8_matmul_rows2(
                first_packed,
                first_scales,
                second_packed,
                second_scales,
                block_count,
                q8_input.row(0),
                input.columns(),
                q8_input.row_scales(0),
                (input_columns + 31) / 32,
                token_count,
                first_output,
                first_output_stride,
                second_output,
                second_output_stride);
        }
        else
        {
            mxfp4_matmul_rows2(
                first_packed,
                first_scales,
                second_packed,
                second_scales,
                block_count,
                float_input,
                input_stride,
                token_count,
                first_output,
                first_output_stride,
                second_output,
                second_output_stride);
        }
    };
    const auto matmul_row_pairs = [&](const uint8_t* packed,
                                      const uint8_t* scales,
                                      uint32_t block_count,
                                      uint32_t row_pair_count,
                                      const float* float_input,
                                      size_t input_stride,
                                      size_t token_count,
                                      float* first_output,
                                      size_t first_pair_stride,
                                      size_t first_token_stride,
                                      float* second_output,
                                      size_t second_pair_stride,
                                      size_t second_token_stride) {
        if (use_q8)
        {
            mxfp4_q8_matmul_row_pairs(
                packed,
                scales,
                block_count,
                row_pair_count,
                q8_input.row(0),
                input.columns(),
                q8_input.row_scales(0),
                (input_columns + 31) / 32,
                token_count,
                first_output,
                first_pair_stride,
                first_token_stride,
                second_output,
                second_pair_stride,
                second_token_stride);
        }
        else
        {
            mxfp4_matmul_row_pairs(
                packed,
                scales,
                block_count,
                row_pair_count,
                float_input,
                input_stride,
                token_count,
                first_output,
                first_pair_stride,
                first_token_stride,
                second_output,
                second_pair_stride,
                second_token_stride);
        }
    };

    // Keep the row-pair producer and apply the activation epilogue in chunks.
    if (input.rows() == 1 && cpu_mxfp4_bulk_row_pairs_enabled(optimization_flags))
    {
        std::vector<float> linear(intermediate_size);
        const uint32_t pair_chunk_size = 16;
        const int64_t pair_group_count = (static_cast<int64_t>(intermediate_size) + pair_chunk_size - 1)
                                         / pair_chunk_size;
#pragma omp parallel for if (allow_openmp_parallel_region())
        for (int64_t pair_group = 0; pair_group < pair_group_count; ++pair_group)
        {
            const uint32_t first_column = static_cast<uint32_t>(pair_group) * pair_chunk_size;
            const uint32_t pair_count = std::min(pair_chunk_size, intermediate_size - first_column);
            const size_t first_row = static_cast<size_t>(first_column) * 2;
            matmul_row_pairs(
                matrix.mxfp4_blocks.data() + first_row * input_columns / 2,
                matrix.mxfp4_scales.data() + first_row * blocks_per_row,
                blocks_per_row,
                pair_count,
                input.row(0),
                input.columns(),
                1,
                output.row(0) + first_column,
                1,
                output.columns(),
                linear.data() + first_column,
                1,
                1);

            if (cpu_fast_silu_enabled(optimization_flags)
                && bias == nullptr
                && activation_limit <= 0.0f
                && (activation == ExpertActivation::Silu
                    || activation == ExpertActivation::DeepSeekSwiGlu
                    || activation == ExpertActivation::GptOssSwiGlu))
            {
                const float sigmoid_scale = activation == ExpertActivation::GptOssSwiGlu ? 1.702f : 1.0f;
                const float up_offset = activation == ExpertActivation::GptOssSwiGlu ? 1.0f : 0.0f;
                float_silu_mul(
                    output.row(0) + first_column,
                    output.row(0) + first_column,
                    linear.data() + first_column,
                    sigmoid_scale,
                    up_offset,
                    pair_count);
                continue;
            }

            for (uint32_t local_column = 0; local_column < pair_count;
                 ++local_column)
            {
                const uint32_t column = first_column + local_column;
                const size_t gate_row = static_cast<size_t>(column) * 2;
                const size_t up_row = gate_row + 1;
                float gate = output.row(0)[column];
                float up = linear[column];
                if (bias)
                {
                    gate += tensor_value(*bias, gate_row);
                    up += tensor_value(*bias, up_row);
                }
                if (activation_limit > 0.0f)
                {
                    gate = std::min(gate, activation_limit);
                    up = std::clamp(up, -activation_limit, activation_limit);
                }
                if (activation == ExpertActivation::Silu
                    || activation == ExpertActivation::DeepSeekSwiGlu)
                {
                    output.row(0)[column] = gate / (1.0f + float_approximate_exp(-gate)) * up;
                }
                else
                {
                    output.row(0)[column] = selected_scaled_silu(gate, 1.702f, approximate_activation)
                                            * (up + 1.0f);
                }
            }
        }
        return output;
    }

    const bool parallel_enabled = allow_openmp_parallel_region();
    const int linear_team_size = parallel_enabled
                                     ? static_cast<int>(cpu_linear_thread_limit())
                                     : 1;
    size_t scratch_worker_count = 1;
#if defined(_OPENMP)
    if (parallel_enabled)
        scratch_worker_count = static_cast<size_t>(linear_team_size);
#endif
    std::vector<float> linear_scratch(
        scratch_worker_count * input.rows());
    const int64_t parallel_columns = static_cast<int64_t>(intermediate_size);
#pragma omp parallel for num_threads(linear_team_size) if (parallel_enabled)
    for (int64_t column = 0; column < parallel_columns; ++column)
    {
        const size_t gate_row = static_cast<size_t>(column) * 2;
        const size_t up_row = gate_row + 1;
        const uint8_t* gate_blocks = matrix.mxfp4_blocks.data() + gate_row * input_columns / 2;
        const uint8_t* up_blocks = matrix.mxfp4_blocks.data() + up_row * input_columns / 2;
        const uint8_t* gate_scales = matrix.mxfp4_scales.data() + gate_row * blocks_per_row;
        const uint8_t* up_scales = matrix.mxfp4_scales.data() + up_row * blocks_per_row;
        const float gate_bias = bias ? tensor_value(*bias, gate_row) : 0.0f;
        const float up_bias = bias ? tensor_value(*bias, up_row) : 0.0f;
        if (input.rows() == 1)
        {
            float gate = 0.0f;
            float linear = 0.0f;
            matmul_rows2(gate_blocks, gate_scales, up_blocks, up_scales, blocks_per_row, input.row(0), input.columns(), 1, &gate, 1, &linear, 1);
            gate += gate_bias;
            linear += up_bias;
            if (activation_limit > 0.0f)
            {
                gate = std::min(gate, activation_limit);
                linear = std::clamp(linear, -activation_limit, activation_limit);
            }
            if (activation == ExpertActivation::Silu
                || activation == ExpertActivation::DeepSeekSwiGlu)
            {
                const float silu = gate / (1.0f + float_approximate_exp(-gate));
                output.row(0)[column] = silu * linear;
            }
            else
            {
                const float silu = selected_scaled_silu(gate, 1.702f, approximate_activation);
                output.row(0)[column] = silu * (linear + 1.0f);
            }
        }
        else
        {
            size_t scratch_worker = 0;
#if defined(_OPENMP)
            if (parallel_enabled)
                scratch_worker = static_cast<size_t>(omp_get_thread_num());
#endif
            float* linear = linear_scratch.data() + scratch_worker * input.rows();
            matmul_rows2(gate_blocks, gate_scales, up_blocks, up_scales, blocks_per_row, input.row(0), input.columns(), input.rows(),
                         output.row(0) + column, output.columns(), linear, 1);
            for (size_t token_index = 0; token_index < input.rows(); ++token_index)
            {
                float gate = output.row(token_index)[column] + gate_bias;
                float up = linear[token_index] + up_bias;
                if (activation_limit > 0.0f)
                {
                    gate = std::min(gate, activation_limit);
                    up = std::clamp(up, -activation_limit, activation_limit);
                }
                if (activation == ExpertActivation::Silu
                    || activation == ExpertActivation::DeepSeekSwiGlu)
                {
                    const float silu = gate / (1.0f + float_approximate_exp(-gate));
                    output.row(token_index)[column] = silu * up;
                }
                else
                {
                    const float silu = selected_scaled_silu(gate, 1.702f, approximate_activation);
                    output.row(token_index)[column] = silu * (up + 1.0f);
                }
            }
        }
    }
    return output;
}

static void locate_mxfp4_group(uint64_t flat_index, std::span<const Mxfp4Task> tasks, bool gate_stage, uint32_t pair_group_size, bool uniform,
                               uint64_t groups_per_task, size_t& task_index, uint32_t& local_index)
{
    if (uniform && groups_per_task != 0)
    {
        task_index = static_cast<size_t>(flat_index / groups_per_task);
        local_index = static_cast<uint32_t>(flat_index % groups_per_task);
        if (task_index < tasks.size())
            return;
    }

    uint64_t offset = 0;
    for (task_index = 0; task_index < tasks.size(); ++task_index)
    {
        const TensorData& matrix = gate_stage ? *tasks[task_index].gate_up : *tasks[task_index].down;
        const uint64_t work_items = gate_stage ? ((matrix.shape[0] / 2) + pair_group_size - 1) / pair_group_size
                                               : (((matrix.shape[0] + 1) / 2) + pair_group_size - 1) / pair_group_size;
        if (flat_index < offset + work_items)
        {
            local_index = static_cast<uint32_t>(flat_index - offset);
            return;
        }
        offset += work_items;
    }
    local_index = 0;
}

static size_t find_shared_q8_input_owner(
    std::span<const Mxfp4Task> tasks,
    std::span<const size_t> owners,
    size_t task_index,
    size_t invalid_owner) noexcept
{
    const Mxfp4Task& task = tasks[task_index];
    if (!task.input)
        return task_index;

    const size_t row_bytes = task.input->rows() == 1
                                 ? static_cast<size_t>(task.input->columns()) * sizeof(float)
                                 : 0;
    for (size_t previous = 0; previous < task_index; ++previous)
    {
        if (owners[previous] == invalid_owner || !tasks[previous].input)
            continue;
        const CpuBatch& candidate = *tasks[previous].input;
        if (&candidate == task.input)
            return owners[previous];
        if (task.input->rows() != 1 || candidate.rows() != 1
            || candidate.columns() != task.input->columns())
            continue;
        if (candidate.row(0) == task.input->row(0)
            || std::memcmp(candidate.row(0), task.input->row(0), row_bytes) == 0)
            return owners[previous];
    }
    return task_index;
}

static bool mxfp4_expert_decode(
    std::span<const Mxfp4Task> tasks,
    Mxfp4Scratch* scratch,
    uint64_t optimization_flags)
{
    Mxfp4Scratch local_scratch;
    Mxfp4Scratch& buffers = scratch ? *scratch : local_scratch;
    buffers.activated.resize(tasks.size());
    buffers.packed_gate_up.resize(tasks.size());
    std::vector<CpuBatch>& activated = buffers.activated;
    std::vector<CpuBatch>& packed_gate_up = buffers.packed_gate_up;
    std::vector<Mxfp4Q8Batch> q8_inputs;
    // Keep Q8/packed for batched paths; single-token AVX512 uses row pairs.
    const bool use_q8 = cpu_mxfp4_q8_enabled(optimization_flags)
                        && mxfp4_kernel_kind() != MxFp4KernelKind::X86Avx512;
    const size_t invalid_q8_owner = tasks.size();
    std::vector<size_t> q8_input_owner(tasks.size(), invalid_q8_owner);
    buffers.q8_activated.resize(tasks.size());
    std::vector<Mxfp4Q8Batch>& q8_activated = buffers.q8_activated;
    std::vector<uint8_t> q8_down_enabled(tasks.size(), 0);
    std::vector<uint8_t> q8_gate_packed(tasks.size(), 0);
    std::vector<uint8_t> q8_down_packed(tasks.size(), 0);
    if (use_q8)
        q8_inputs.resize(tasks.size());
    const bool approximate_activation = cpu_fast_silu_enabled(optimization_flags);
    const uint32_t pair_group_size = mxfp4_decode_row_pair_group_size();
    uint64_t gate_pair_group_count = 0;
    uint64_t down_pair_group_count = 0;
    uint64_t uniform_gate_pair_group_count = 0;
    uint64_t uniform_down_pair_group_count = 0;
    bool uniform_gate_pair_groups = true;
    bool uniform_down_pair_groups = true;
    uint64_t operation_count = 0;
    for (size_t task_index = 0; task_index < tasks.size(); ++task_index)
    {
        const Mxfp4Task& task = tasks[task_index];
        if (!task.gate_up || !task.down || !task.input || !task.output || task.input->rows() != 1 || task.gate_up->dtype != DType::MxFp4
            || task.down->dtype != DType::MxFp4 || task.gate_up->shape.size() != 2 || task.down->shape.size() != 2 || task.gate_up->shape[0] % 2 != 0
            || task.gate_up->shape[1] != task.input->columns() || task.down->shape[1] != task.gate_up->shape[0] / 2)
        {
            return false;
        }
        if (task.gate_up_bias && task.gate_up_bias->shape != std::vector<uint32_t>{task.gate_up->shape[0]})
        {
            return false;
        }
        if (task.down_bias && task.down_bias->shape != std::vector<uint32_t>{task.down->shape[0]})
        {
            return false;
        }
        const uint32_t intermediate_size = task.gate_up->shape[0] / 2;
        activated[task_index].reset(1, intermediate_size, false);
        if (use_q8 && intermediate_size % 32 == 0)
        {
            q8_down_enabled[task_index] = 1;
            if (cpu_packed_weights_enabled(optimization_flags)
                && task.down->shape[0] >= 4
                && mxfp4_q8_packed_kernel_available())
            {
                const std::shared_ptr<const Mxfp4Q8PackedMatrix> packed_weights = get_mxfp4_q8_packed_weights(
                    *task.down,
                    intermediate_size / 32,
                    task.down->shape[0]);
                q8_down_packed[task_index] = packed_weights ? 1 : 0;
            }
        }
        if (use_q8 && task.input->columns() % 32 == 0)
        {
            const size_t input_owner = find_shared_q8_input_owner(
                tasks,
                q8_input_owner,
                task_index,
                invalid_q8_owner);
            q8_input_owner[task_index] = input_owner;
            if (input_owner == task_index)
            {
                mxfp4_q8_quantize_batch(
                    task.input->row(0),
                    task.input->columns(),
                    1,
                    task.input->columns(),
                    q8_inputs[input_owner]);
            }
            if (cpu_packed_weights_enabled(optimization_flags)
                && task.gate_up->shape[0] >= 4
                && mxfp4_q8_packed_kernel_available())
            {
                // The fixed Expert team completes the sidecar GEMV below.
                q8_gate_packed[task_index] = 1;
            }
        }
        task.output->reset(1, task.down->shape[0], false);
        const uint64_t task_gate_pair_groups = (intermediate_size + pair_group_size - 1) / pair_group_size;
        const uint64_t task_down_pair_groups = ((task.down->shape[0] + 1) / 2 + pair_group_size - 1) / pair_group_size;
        gate_pair_group_count += task_gate_pair_groups;
        down_pair_group_count += task_down_pair_groups;
        if (task_index == 0)
        {
            uniform_gate_pair_group_count = task_gate_pair_groups;
            uniform_down_pair_group_count = task_down_pair_groups;
        }
        else
        {
            uniform_gate_pair_groups = uniform_gate_pair_groups && task_gate_pair_groups == uniform_gate_pair_group_count;
            uniform_down_pair_groups = uniform_down_pair_groups && task_down_pair_groups == uniform_down_pair_group_count;
        }
        const uint64_t gate_operations = static_cast<uint64_t>(task.gate_up->shape[0]) * task.gate_up->shape[1];
        const uint64_t down_operations = static_cast<uint64_t>(task.down->shape[0]) * task.down->shape[1];
        const uint64_t maximum_operations = std::numeric_limits<uint64_t>::max();
        if (gate_operations > maximum_operations - down_operations)
        {
            operation_count = std::numeric_limits<uint64_t>::max();
        }
        else
        {
            const uint64_t task_operations = gate_operations + down_operations;
            if (operation_count > maximum_operations - task_operations)
            {
                operation_count = maximum_operations;
            }
            else
            {
                operation_count += task_operations;
            }
        }
    }

    const int group_team_size = openmp_mxfp4_group_team_size(operation_count);
    const bool parallelize_group = group_team_size > 1 && allow_openmp_parallel_region();
    const int64_t parallel_gate_pair_groups = static_cast<int64_t>(gate_pair_group_count);
    const int64_t parallel_down_pair_groups = static_cast<int64_t>(down_pair_group_count);
#pragma omp parallel num_threads(group_team_size) if (parallelize_group)
    {
#pragma omp for schedule(static)
        for (int64_t task_index = 0;
             task_index < static_cast<int64_t>(tasks.size());
             ++task_index)
        {
            if (!q8_gate_packed[static_cast<size_t>(task_index)])
                continue;
            const Mxfp4Task& task = tasks[static_cast<size_t>(task_index)];
            const size_t input_owner = q8_input_owner[static_cast<size_t>(task_index)];
            const std::shared_ptr<const Mxfp4Q8PackedMatrix> packed_weights = get_mxfp4_q8_packed_weights(
                *task.gate_up,
                task.input->columns() / 32,
                task.gate_up->shape[0]);
            if (!packed_weights)
            {
                q8_gate_packed[static_cast<size_t>(task_index)] = 0;
                continue;
            }
            packed_gate_up[static_cast<size_t>(task_index)].reset(
                1,
                task.gate_up->shape[0],
                false);
            mxfp4_q8_packed_gemv(
                *packed_weights,
                q8_inputs[input_owner].row(0),
                q8_inputs[input_owner].row_scales(0),
                packed_gate_up[static_cast<size_t>(task_index)].row(0));
        }

#pragma omp for schedule(static)
        for (int64_t flat_group = 0; flat_group < parallel_gate_pair_groups; ++flat_group)
        {
            size_t task_index = 0;
            uint32_t pair_group = 0;
            locate_mxfp4_group(static_cast<uint64_t>(flat_group), tasks, true, pair_group_size, uniform_gate_pair_groups, uniform_gate_pair_group_count,
                               task_index, pair_group);
            const Mxfp4Task& task = tasks[task_index];
            const TensorData& matrix = *task.gate_up;
            const uint32_t input_columns = matrix.shape[1];
            const uint32_t blocks_per_row = input_columns / 32;
            const uint32_t intermediate_size = matrix.shape[0] / 2;
            const uint32_t first_column = pair_group * pair_group_size;
            const uint32_t column_count = std::min<uint32_t>(pair_group_size, intermediate_size - first_column);
            const size_t gate_row = static_cast<size_t>(first_column) * 2;
            const size_t input_owner = q8_input_owner[task_index];
            float gates[2] = {};
            float linears[2] = {};
            if (q8_gate_packed[task_index])
            {
                for (uint32_t local_column = 0; local_column < column_count; ++local_column)
                {
                    const uint32_t column = first_column + local_column;
                    gates[local_column] = packed_gate_up[task_index].row(0)[column * 2];
                    linears[local_column] = packed_gate_up[task_index].row(0)[column * 2 + 1];
                }
            }
            else if (use_q8 && input_columns % 32 == 0)
            {
                mxfp4_q8_matmul_row_pairs(
                    matrix.mxfp4_blocks.data() + gate_row * input_columns / 2,
                    matrix.mxfp4_scales.data() + gate_row * blocks_per_row,
                    blocks_per_row,
                    column_count,
                    q8_inputs[input_owner].row(0),
                    input_columns,
                    q8_inputs[input_owner].row_scales(0),
                    (input_columns + 31) / 32,
                    1,
                    gates,
                    1,
                    1,
                    linears,
                    1,
                    1);
            }
            else
            {
                mxfp4_matmul_row_pairs(
                    matrix.mxfp4_blocks.data() + gate_row * input_columns / 2,
                    matrix.mxfp4_scales.data() + gate_row * blocks_per_row,
                    blocks_per_row,
                    column_count,
                    task.input->row(0),
                    task.input->columns(),
                    1,
                    gates,
                    1,
                    1,
                    linears,
                    1,
                    1);
            }
            for (uint32_t local_column = 0; local_column < column_count; ++local_column)
            {
                const uint32_t column = first_column + local_column;
                const size_t local_gate_row = static_cast<size_t>(column) * 2;
                const size_t local_up_row = local_gate_row + 1;
                float gate = gates[local_column];
                float linear = linears[local_column];
                if (task.gate_up_bias)
                {
                    gate += tensor_value(*task.gate_up_bias, local_gate_row);
                    linear += tensor_value(*task.gate_up_bias, local_up_row);
                }
                if (task.activation_limit > 0.0f)
                {
                    gate = std::min(gate, task.activation_limit);
                    linear = std::clamp(linear, -task.activation_limit, task.activation_limit);
                }
                if (task.activation == ExpertActivation::Silu
                    || task.activation == ExpertActivation::DeepSeekSwiGlu)
                {
                    const float silu = gate / (1.0f + float_approximate_exp(-gate));
                    activated[task_index].row(0)[column] = silu * linear;
                }
                else
                {
                    const float silu = selected_scaled_silu(gate, 1.702f, approximate_activation);
                    activated[task_index].row(0)[column] = silu * (linear + 1.0f);
                }
            }
        }

        if (use_q8)
        {
#pragma omp for schedule(static)
            for (int64_t task_index = 0;
                 task_index < static_cast<int64_t>(tasks.size());
                 ++task_index)
            {
                if (q8_down_enabled[static_cast<size_t>(task_index)])
                {
                    const CpuBatch& task_activated = activated[static_cast<size_t>(task_index)];
                    mxfp4_q8_quantize_batch(
                        task_activated.row(0),
                        task_activated.columns(),
                        task_activated.rows(),
                        task_activated.columns(),
                        q8_activated[static_cast<size_t>(task_index)]);
                }
            }
        }

#pragma omp for schedule(static)
        for (int64_t task_index = 0;
             task_index < static_cast<int64_t>(tasks.size());
             ++task_index)
        {
            if (!q8_down_packed[static_cast<size_t>(task_index)])
                continue;
            const Mxfp4Task& task = tasks[static_cast<size_t>(task_index)];
            const std::shared_ptr<const Mxfp4Q8PackedMatrix> packed_weights = get_mxfp4_q8_packed_weights(
                *task.down,
                task.down->shape[1] / 32,
                task.down->shape[0]);
            if (!packed_weights)
            {
                q8_down_packed[static_cast<size_t>(task_index)] = 0;
                continue;
            }
            mxfp4_q8_packed_gemv(
                *packed_weights,
                q8_activated[static_cast<size_t>(task_index)].row(0),
                q8_activated[static_cast<size_t>(task_index)].row_scales(0),
                task.output->row(0));
        }

#pragma omp for schedule(static)
        for (int64_t flat_group = 0; flat_group < parallel_down_pair_groups; ++flat_group)
        {
            size_t task_index = 0;
            uint32_t pair_group = 0;
            locate_mxfp4_group(static_cast<uint64_t>(flat_group), tasks, false, pair_group_size, uniform_down_pair_groups, uniform_down_pair_group_count,
                               task_index, pair_group);
            const Mxfp4Task& task = tasks[task_index];
            const TensorData& matrix = *task.down;
            const uint32_t input_columns = matrix.shape[1];
            const uint32_t blocks_per_row = input_columns / 32;
            const uint32_t first_pair = pair_group * pair_group_size;
            const uint32_t full_pair_count = matrix.shape[0] / 2;
            const uint32_t pair_count = first_pair < full_pair_count ? std::min<uint32_t>(pair_group_size, full_pair_count - first_pair) : 0;
            const uint32_t first_row = first_pair * 2;
            float* first_output = task.output->row(0) + first_row;
            if (pair_count != 0)
            {
                if (q8_down_packed[task_index])
                {
                    // The complete down projection was written above.
                }
                else if (q8_down_enabled[task_index])
                {
                    mxfp4_q8_matmul_row_pairs(
                        matrix.mxfp4_blocks.data() + static_cast<size_t>(first_row) * input_columns / 2,
                        matrix.mxfp4_scales.data() + static_cast<size_t>(first_row) * blocks_per_row,
                        blocks_per_row,
                        pair_count,
                        q8_activated[task_index].row(0),
                        activated[task_index].columns(),
                        q8_activated[task_index].row_scales(0),
                        (activated[task_index].columns() + 31) / 32,
                        1,
                        first_output,
                        2,
                        1,
                        first_output + 1,
                        2,
                        1);
                }
                else
                {
                    mxfp4_matmul_row_pairs(matrix.mxfp4_blocks.data() + static_cast<size_t>(first_row) * input_columns / 2,
                                           matrix.mxfp4_scales.data() + static_cast<size_t>(first_row) * blocks_per_row, blocks_per_row, pair_count,
                                           activated[task_index].row(0), activated[task_index].columns(), 1, first_output, 2, 1, first_output + 1, 2, 1);
                }
                if (task.down_bias)
                {
                    for (uint32_t row = 0; row < pair_count * 2; ++row)
                    {
                        first_output[row] += tensor_value(*task.down_bias, first_row + row);
                    }
                }
            }
            const uint32_t odd_row = full_pair_count * 2;
            if (odd_row < matrix.shape[0] && first_pair + pair_count == full_pair_count)
            {
                float* odd_output = task.output->row(0) + odd_row;
                if (q8_down_packed[task_index])
                {
                    // The packed GEMV already produced the odd output row.
                }
                else if (q8_down_enabled[task_index])
                {
                    odd_output[0] = mxfp4_q8_dot(
                        matrix.mxfp4_blocks.data() + static_cast<size_t>(odd_row) * input_columns / 2,
                        matrix.mxfp4_scales.data() + static_cast<size_t>(odd_row) * blocks_per_row,
                        blocks_per_row,
                        q8_activated[task_index].row(0),
                        q8_activated[task_index].row_scales(0));
                }
                else
                {
                    odd_output[0] = mxfp4_dot(
                        matrix.mxfp4_blocks.data() + static_cast<size_t>(odd_row) * input_columns / 2,
                        matrix.mxfp4_scales.data() + static_cast<size_t>(odd_row) * blocks_per_row, blocks_per_row, activated[task_index].row(0));
                }
                if (task.down_bias)
                {
                    odd_output[0] += tensor_value(*task.down_bias, odd_row);
                }
            }
        }
    }
    return true;
}

static uint64_t exact_float_row_hash(
    const float* values,
    uint32_t count) noexcept
{
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint32_t index = 0; index < count; ++index)
    {
        uint32_t bits = 0;
        std::memcpy(
            &bits,
            values + index,
            sizeof(bits));
        hash ^= bits;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool mxfp4_expert_batch(std::span<const Mxfp4Task> tasks, Mxfp4Scratch* scratch, uint64_t optimization_flags)
{
    if (tasks.empty())
        return true;
    bool single_token = true;
    for (const Mxfp4Task& task : tasks)
    {
        if (task.activation != ExpertActivation::Silu
            && task.activation != ExpertActivation::GptOssSwiGlu
            && task.activation != ExpertActivation::DeepSeekSwiGlu)
            return false;
        if (!task.input || task.input->rows() != 1)
            single_token = false;
    }
    Mxfp4Scratch local_scratch;
    Mxfp4Scratch& buffers = scratch ? *scratch : local_scratch;
    if (single_token)
    {
        if (scratch)
        {
            buffers.physical_input_rows.assign(
                tasks.size(),
                1);
            buffers.unique_row_maps.resize(tasks.size());
            for (std::vector<uint32_t>& row_map :
                 buffers.unique_row_maps)
            {
                row_map.clear();
            }
        }
        return mxfp4_expert_decode(tasks, scratch, optimization_flags);
    }

    buffers.unique_input.resize(tasks.size());
    buffers.unique_output.resize(tasks.size());
    buffers.unique_row_maps.resize(tasks.size());
    buffers.effective_tasks.assign(
        tasks.begin(),
        tasks.end());
    buffers.physical_input_rows.resize(tasks.size());
    for (size_t task_index = 0;
         task_index < tasks.size();
         ++task_index)
    {
        const Mxfp4Task& task = tasks[task_index];
        std::vector<uint32_t>& row_map = buffers.unique_row_maps[task_index];
        row_map.clear();
        if (!task.input || task.input->rows() <= 1)
        {
            buffers.physical_input_rows[task_index] = task.input
                                                          ? static_cast<uint32_t>(task.input->rows())
                                                          : 0;
            continue;
        }

        std::vector<uint32_t> representatives;
        std::vector<uint64_t> representative_hashes;
        representatives.reserve(task.input->rows());
        representative_hashes.reserve(task.input->rows());
        row_map.resize(task.input->rows());
        const size_t row_bytes = static_cast<size_t>(task.input->columns())
                                 * sizeof(float);
        for (size_t row = 0;
             row < task.input->rows();
             ++row)
        {
            uint32_t selected = static_cast<uint32_t>(
                representatives.size());
            const uint64_t row_hash = exact_float_row_hash(
                task.input->row(row),
                task.input->columns());
            for (size_t unique_row = 0;
                 unique_row < representatives.size();
                 ++unique_row)
            {
                if (representative_hashes[unique_row]
                        == row_hash
                    && std::memcmp(
                           task.input->row(row),
                           task.input->row(
                               representatives[unique_row]),
                           row_bytes)
                           == 0)
                {
                    selected = static_cast<uint32_t>(unique_row);
                    break;
                }
            }
            if (selected == representatives.size())
            {
                representatives.push_back(
                    static_cast<uint32_t>(row));
                representative_hashes.push_back(row_hash);
            }
            row_map[row] = selected;
        }
        buffers.physical_input_rows[task_index] = static_cast<uint32_t>(
            representatives.size());
        if (representatives.size()
            == task.input->rows())
        {
            row_map.clear();
            continue;
        }

        CpuBatch& unique_input = buffers.unique_input[task_index];
        unique_input.reset(
            representatives.size(),
            task.input->columns(),
            false);
        for (size_t unique_row = 0;
             unique_row < representatives.size();
             ++unique_row)
        {
            std::copy_n(
                task.input->row(
                    representatives[unique_row]),
                task.input->columns(),
                unique_input.row(unique_row));
        }
        buffers.effective_tasks[task_index].input = &unique_input;
        buffers.effective_tasks[task_index].output = &buffers.unique_output[task_index];
    }
    const std::span<const Mxfp4Task> execution_tasks = buffers.effective_tasks;
    buffers.activated.resize(tasks.size());
    buffers.linear.resize(tasks.size());
    const bool use_q8 = cpu_mxfp4_q8_enabled(optimization_flags);
    std::vector<Mxfp4Q8Batch> q8_inputs;
    const size_t invalid_q8_owner = execution_tasks.size();
    std::vector<size_t> q8_input_owner(execution_tasks.size(), invalid_q8_owner);
    std::vector<uint8_t> q8_gate_enabled(execution_tasks.size(), 0);
    std::vector<uint8_t> q8_down_enabled(execution_tasks.size(), 0);
    if (use_q8)
        q8_inputs.resize(execution_tasks.size());
    buffers.q8_activated.resize(tasks.size());
    std::vector<CpuBatch>& activated = buffers.activated;
    std::vector<CpuBatch>& linear = buffers.linear;
    std::vector<Mxfp4Q8Batch>& q8_activated = buffers.q8_activated;
    const bool approximate_activation = cpu_fast_silu_enabled(optimization_flags);
    uint64_t gate_column_count = 0;
    uint64_t down_row_pair_count = 0;
    uint64_t operation_count = 0;
    for (size_t task_index = 0; task_index < execution_tasks.size(); ++task_index)
    {
        const Mxfp4Task& task = execution_tasks[task_index];
        if (!task.gate_up || !task.down || !task.input || !task.output || task.input->rows() == 0 || task.gate_up->dtype != DType::MxFp4
            || task.down->dtype != DType::MxFp4 || task.gate_up->shape.size() != 2 || task.down->shape.size() != 2 || task.gate_up->shape[0] % 2 != 0
            || task.gate_up->shape[1] != task.input->columns() || task.down->shape[1] != task.gate_up->shape[0] / 2)
        {
            return false;
        }
        if (task.gate_up_bias && task.gate_up_bias->shape != std::vector<uint32_t>{task.gate_up->shape[0]})
        {
            return false;
        }
        if (task.down_bias && task.down_bias->shape != std::vector<uint32_t>{task.down->shape[0]})
        {
            return false;
        }
        const uint32_t intermediate_size = task.gate_up->shape[0] / 2;
        activated[task_index].reset(task.input->rows(), intermediate_size, false);
        linear[task_index].reset(task.input->rows(), intermediate_size, false);
        if (use_q8 && task.input->columns() % 32 == 0)
        {
            const size_t input_owner = find_shared_q8_input_owner(
                execution_tasks,
                q8_input_owner,
                task_index,
                invalid_q8_owner);
            q8_input_owner[task_index] = input_owner;
            if (input_owner == task_index)
            {
                mxfp4_q8_quantize_batch(
                    task.input->row(0),
                    task.input->columns(),
                    task.input->rows(),
                    task.input->columns(),
                    q8_inputs[input_owner]);
            }
            q8_gate_enabled[task_index] = 1;
        }
        if (use_q8 && intermediate_size % 32 == 0)
            q8_down_enabled[task_index] = 1;
        task.output->reset(task.input->rows(), task.down->shape[0], false);
        gate_column_count += intermediate_size;
        down_row_pair_count += (task.down->shape[0] + 1) / 2;
        const uint64_t gate_operations = static_cast<uint64_t>(task.gate_up->shape[0]) * task.gate_up->shape[1];
        const uint64_t down_operations = static_cast<uint64_t>(task.down->shape[0]) * task.down->shape[1];
        const uint64_t maximum_operations = std::numeric_limits<uint64_t>::max();
        if (gate_operations > maximum_operations - down_operations)
        {
            operation_count = maximum_operations;
        }
        else
        {
            const uint64_t task_operations = gate_operations + down_operations;
            if (operation_count > maximum_operations - task_operations)
            {
                operation_count = maximum_operations;
            }
            else
            {
                operation_count += task_operations;
            }
        }
    }

    const int group_team_size = openmp_mxfp4_group_team_size(operation_count);
    const bool parallelize_group = group_team_size > 1 && allow_openmp_parallel_region();
#pragma omp parallel num_threads(group_team_size) if (parallelize_group)
    {
        uint64_t thread_index = 0;
        uint64_t actual_team_size = 1;
#if defined(_OPENMP)
        thread_index = static_cast<uint64_t>(omp_get_thread_num());
        actual_team_size = static_cast<uint64_t>(omp_get_num_threads());
#endif
        const uint64_t gate_begin = gate_column_count * thread_index / actual_team_size;
        const uint64_t gate_end = gate_column_count * (thread_index + 1) / actual_team_size;
        uint64_t task_begin = 0;
        for (size_t task_index = 0; task_index < execution_tasks.size(); ++task_index)
        {
            const Mxfp4Task& task = execution_tasks[task_index];
            const TensorData& matrix = *task.gate_up;
            const uint64_t task_end = task_begin + matrix.shape[0] / 2;
            const uint64_t intersection_begin = std::max(gate_begin, task_begin);
            const uint64_t intersection_end = std::min(gate_end, task_end);
            if (intersection_begin >= intersection_end)
            {
                task_begin = task_end;
                continue;
            }
            const uint64_t local_begin = intersection_begin - task_begin;
            const uint64_t local_end = intersection_end - task_begin;
            const uint32_t input_columns = matrix.shape[1];
            const uint32_t blocks_per_row = input_columns / 32;
            const size_t first_row = static_cast<size_t>(local_begin) * 2;
            if (q8_gate_enabled[task_index])
            {
                const size_t input_owner = q8_input_owner[task_index];
                mxfp4_q8_matmul_row_pairs(
                    matrix.mxfp4_blocks.data() + first_row * input_columns / 2,
                    matrix.mxfp4_scales.data() + first_row * blocks_per_row,
                    blocks_per_row,
                    static_cast<uint32_t>(local_end - local_begin),
                    q8_inputs[input_owner].row(0),
                    task.input->columns(),
                    q8_inputs[input_owner].row_scales(0),
                    (task.input->columns() + 31) / 32,
                    task.input->rows(),
                    activated[task_index].row(0) + local_begin,
                    1,
                    activated[task_index].columns(),
                    linear[task_index].row(0) + local_begin,
                    1,
                    linear[task_index].columns());
            }
            else
            {
                mxfp4_matmul_row_pairs(
                    matrix.mxfp4_blocks.data() + first_row * input_columns / 2,
                    matrix.mxfp4_scales.data() + first_row * blocks_per_row,
                    blocks_per_row,
                    static_cast<uint32_t>(local_end - local_begin),
                    task.input->row(0),
                    task.input->columns(),
                    task.input->rows(),
                    activated[task_index].row(0) + local_begin,
                    1,
                    activated[task_index].columns(),
                    linear[task_index].row(0) + local_begin,
                    1,
                    linear[task_index].columns());
            }
            for (size_t token_index = 0; token_index < task.input->rows(); ++token_index)
            {
                float* gate_values = activated[task_index].row(token_index);
                float* linear_values = linear[task_index].row(token_index);
                if (cpu_fast_silu_enabled(optimization_flags)
                    && task.gate_up_bias == nullptr
                    && task.activation_limit <= 0.0f
                    && (task.activation == ExpertActivation::Silu
                        || task.activation == ExpertActivation::DeepSeekSwiGlu
                        || task.activation == ExpertActivation::GptOssSwiGlu))
                {
                    const float sigmoid_scale = task.activation == ExpertActivation::GptOssSwiGlu ? 1.702f : 1.0f;
                    const float up_offset = task.activation == ExpertActivation::GptOssSwiGlu ? 1.0f : 0.0f;
                    float_silu_mul(
                        gate_values + local_begin,
                        gate_values + local_begin,
                        linear_values + local_begin,
                        sigmoid_scale,
                        up_offset,
                        static_cast<uint32_t>(local_end - local_begin));
                    continue;
                }
                for (uint64_t column = local_begin; column < local_end; ++column)
                {
                    const size_t gate_row = static_cast<size_t>(column) * 2;
                    const size_t up_row = gate_row + 1;
                    float gate = gate_values[column];
                    float up = linear_values[column];
                    if (task.gate_up_bias)
                    {
                        gate += tensor_value(*task.gate_up_bias, gate_row);
                        up += tensor_value(*task.gate_up_bias, up_row);
                    }
                    if (task.activation_limit > 0.0f)
                    {
                        gate = std::min(gate, task.activation_limit);
                        up = std::clamp(up, -task.activation_limit, task.activation_limit);
                    }
                    if (task.activation == ExpertActivation::Silu
                        || task.activation
                               == ExpertActivation::DeepSeekSwiGlu)
                    {
                        const float silu = gate
                                           / (1.0f + float_approximate_exp(-gate));
                        gate_values[column] = silu * up;
                    }
                    else
                    {
                        const float silu = selected_scaled_silu(
                            gate,
                            1.702f,
                            approximate_activation);
                        gate_values[column] = silu * (up + 1.0f);
                    }
                }
            }
            task_begin = task_end;
        }
    }

    if (use_q8)
    {
#pragma omp parallel for num_threads(group_team_size) if (parallelize_group) schedule(static)
        for (int64_t task_index = 0;
             task_index < static_cast<int64_t>(execution_tasks.size());
             ++task_index)
        {
            if (q8_down_enabled[static_cast<size_t>(task_index)])
            {
                const CpuBatch& task_activated = activated[static_cast<size_t>(task_index)];
                mxfp4_q8_quantize_batch(
                    task_activated.row(0),
                    task_activated.columns(),
                    task_activated.rows(),
                    task_activated.columns(),
                    q8_activated[static_cast<size_t>(task_index)]);
            }
        }
    }

#pragma omp parallel num_threads(group_team_size) if (parallelize_group)
    {
        uint64_t thread_index = 0;
        uint64_t actual_team_size = 1;
#if defined(_OPENMP)
        thread_index = static_cast<uint64_t>(omp_get_thread_num());
        actual_team_size = static_cast<uint64_t>(omp_get_num_threads());
#endif
        const uint64_t down_begin = down_row_pair_count * thread_index / actual_team_size;
        const uint64_t down_end = down_row_pair_count * (thread_index + 1) / actual_team_size;
        uint64_t task_begin = 0;
        for (size_t task_index = 0; task_index < execution_tasks.size(); ++task_index)
        {
            const Mxfp4Task& task = execution_tasks[task_index];
            const TensorData& matrix = *task.down;
            const uint64_t task_pair_count = matrix.shape[0] / 2;
            const uint64_t task_end = task_begin + (matrix.shape[0] + 1) / 2;
            const uint64_t intersection_begin = std::max(down_begin, task_begin);
            const uint64_t intersection_end = std::min(down_end, task_end);
            const uint64_t local_begin = intersection_begin > task_begin ? intersection_begin - task_begin : 0;
            const uint64_t local_end = intersection_end > task_begin ? std::min<uint64_t>(task_pair_count, intersection_end - task_begin) : 0;
            if (intersection_begin >= intersection_end || local_begin >= local_end)
            {
                task_begin = task_end;
                continue;
            }
            const uint32_t input_columns = matrix.shape[1];
            const uint32_t blocks_per_row = input_columns / 32;
            const size_t first_row = static_cast<size_t>(local_begin) * 2;
            if (q8_down_enabled[task_index])
            {
                mxfp4_q8_matmul_row_pairs(
                    matrix.mxfp4_blocks.data() + first_row * input_columns / 2,
                    matrix.mxfp4_scales.data() + first_row * blocks_per_row,
                    blocks_per_row,
                    static_cast<uint32_t>(local_end - local_begin),
                    q8_activated[task_index].row(0),
                    activated[task_index].columns(),
                    q8_activated[task_index].row_scales(0),
                    (activated[task_index].columns() + 31) / 32,
                    activated[task_index].rows(),
                    task.output->row(0) + first_row,
                    2,
                    task.output->columns(),
                    task.output->row(0) + first_row + 1,
                    2,
                    task.output->columns());
            }
            else
            {
                mxfp4_matmul_row_pairs(
                    matrix.mxfp4_blocks.data() + first_row * input_columns / 2,
                    matrix.mxfp4_scales.data() + first_row * blocks_per_row,
                    blocks_per_row,
                    static_cast<uint32_t>(local_end - local_begin),
                    activated[task_index].row(0),
                    activated[task_index].columns(),
                    activated[task_index].rows(),
                    task.output->row(0) + first_row,
                    2,
                    task.output->columns(),
                    task.output->row(0) + first_row + 1,
                    2,
                    task.output->columns());
            }
            if (task.down_bias)
            {
                for (size_t token_index = 0; token_index < task.output->rows(); ++token_index)
                {
                    float* output = task.output->row(token_index);
                    for (uint64_t row_pair = local_begin; row_pair < local_end; ++row_pair)
                    {
                        const size_t row = static_cast<size_t>(row_pair) * 2;
                        output[row] += tensor_value(*task.down_bias, row);
                        output[row + 1] += tensor_value(*task.down_bias, row + 1);
                    }
                }
            }
            task_begin = task_end;
        }
    }

    for (size_t task_index = 0; task_index < execution_tasks.size(); ++task_index)
    {
        const Mxfp4Task& task = execution_tasks[task_index];
        const TensorData& matrix = *task.down;
        if (matrix.shape[0] % 2 == 0)
            continue;
        const uint32_t row = matrix.shape[0] - 1;
        const uint32_t blocks_per_row = matrix.shape[1] / 32;
        if (q8_down_enabled[task_index])
        {
            mxfp4_q8_gemm_row(
                matrix.mxfp4_blocks.data() + static_cast<size_t>(row) * matrix.shape[1] / 2,
                matrix.mxfp4_scales.data() + static_cast<size_t>(row) * blocks_per_row,
                blocks_per_row,
                q8_activated[task_index].row(0),
                activated[task_index].columns(),
                q8_activated[task_index].row_scales(0),
                (activated[task_index].columns() + 31) / 32,
                activated[task_index].rows(),
                task.output->row(0) + row,
                task.output->columns());
        }
        else
        {
            mxfp4_gemm_row(
                matrix.mxfp4_blocks.data() + static_cast<size_t>(row) * matrix.shape[1] / 2,
                matrix.mxfp4_scales.data() + static_cast<size_t>(row) * blocks_per_row,
                blocks_per_row,
                activated[task_index].row(0),
                activated[task_index].columns(),
                activated[task_index].rows(),
                task.output->row(0) + row,
                task.output->columns());
        }
        if (task.down_bias)
        {
            const float bias = tensor_value(*task.down_bias, row);
            for (size_t token_index = 0; token_index < task.output->rows(); ++token_index)
            {
                task.output->row(token_index)[row] += bias;
            }
        }
    }
    for (size_t task_index = 0;
         task_index < tasks.size();
         ++task_index)
    {
        const std::vector<uint32_t>& row_map = buffers.unique_row_maps[task_index];
        if (row_map.empty())
            continue;
        const Mxfp4Task& task = tasks[task_index];
        const CpuBatch& unique_output = buffers.unique_output[task_index];
        task.output->reset(
            row_map.size(),
            unique_output.columns(),
            false);
        for (size_t row = 0; row < row_map.size(); ++row)
        {
            std::copy_n(
                unique_output.row(row_map[row]),
                unique_output.columns(),
                task.output->row(row));
        }
    }
    return true;
}

void linear_batch_into(const TensorData& matrix, const TensorData& bias, const CpuBatch& input, CpuBatch& output, uint64_t optimization_flags, const CompiledOperator* executable, ExecutionBackend backend)
{
    if (backend == ExecutionBackend::Vulkan
        && ((executable && executable->bfloat16
             && executable->bfloat16->forward(input, output))
            || (executable && executable->float8 && executable->float8->forward(input, output))
            || (executable && executable->qnk && executable->qnk->forward(input, output))
            || (executable && executable->linear && executable->linear->forward(input, output))))
    {
        return;
    }
    linear_batch_into(matrix, input, output, optimization_flags, executable, backend);
    require_dense_host_storage(bias, "linear bias");
    assert(bias.shape.size() == 1 && bias.shape[0] == output.columns());
    if (bias.dtype == DType::Float32)
    {
        const std::span<const float> values = bias.float32_values();
        for (size_t token_index = 0; token_index < output.rows(); ++token_index)
            float_scaled_add(output.row(token_index), values.data(), 1.0f, output.columns());
        return;
    }
    if (bias.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> values = bias.bfloat16_values();
        for (size_t token_index = 0; token_index < output.rows(); ++token_index)
            bfloat16_scaled_add(output.row(token_index), values.data(), 1.0f, output.columns());
        return;
    }
    for (size_t token_index = 0; token_index < output.rows(); ++token_index)
    {
        float* row = output.row(token_index);
        for (uint32_t column = 0; column < output.columns(); ++column)
            row[column] += tensor_value(bias, column);
    }
}

CpuBatch linear_batch(const TensorData& matrix, const TensorData& bias, const CpuBatch& input, uint64_t optimization_flags, const CompiledOperator* executable, ExecutionBackend backend)
{
    CpuBatch output;
    linear_batch_into(matrix, bias, input, output, optimization_flags, executable, backend);
    return output;
}

void rms_norm_batch_into(const CpuBatch& input, const TensorData& weight, float epsilon, CpuBatch& output, float weight_offset, uint64_t optimization_flags)
{
    assert(weight.element_count() == input.columns());
    require_dense_host_storage(weight, "normalization weight");
    output.reset(input.rows(), input.columns(), false);
    const int64_t row_count = static_cast<int64_t>(input.rows());
#pragma omp parallel for schedule(static) if (row_count > 1 && allow_openmp_parallel_region())
    for (int64_t token = 0; token < row_count; ++token)
    {
        const size_t token_index = static_cast<size_t>(token);
        const float* source = input.row(token_index);
        float* destination = output.row(token_index);
        if (simd_rms_norm_enabled(optimization_flags)
            && (weight.dtype == DType::Float32
                || weight.dtype == DType::BFloat16))
        {
            if (weight.dtype == DType::Float32)
            {
                float_rms_norm(
                    destination,
                    source,
                    weight.float32_values().data(),
                    epsilon,
                    weight_offset,
                    input.columns());
            }
            else
            {
                bfloat16_rms_norm(
                    destination,
                    source,
                    weight.bfloat16_values().data(),
                    epsilon,
                    weight_offset,
                    input.columns());
            }
            continue;
        }
        float sum_of_squares = 0.0f;
        for (uint32_t column = 0; column < input.columns(); ++column)
            sum_of_squares += source[column] * source[column];
        const float inverse_rms = 1.0f / std::sqrt(sum_of_squares / static_cast<float>(input.columns()) + epsilon);
        for (uint32_t column = 0; column < input.columns(); ++column)
            destination[column] = source[column] * inverse_rms * (tensor_value(weight, column) + weight_offset);
    }
}

CpuBatch rms_norm_batch(const CpuBatch& input, const TensorData& weight, float epsilon, float weight_offset, uint64_t optimization_flags)
{
    CpuBatch output;
    rms_norm_batch_into(input, weight, epsilon, output, weight_offset, optimization_flags);
    return output;
}

void add_bias_inplace(CpuBatch& destination, const TensorData& bias)
{
    assert(bias.shape.size() == 1 && bias.shape[0] == destination.columns());
    require_dense_host_storage(bias, "bias");
    if (bias.dtype == DType::Float32)
    {
        const std::span<const float> values = bias.float32_values();
        for (size_t row_index = 0; row_index < destination.rows(); ++row_index)
            float_scaled_add(destination.row(row_index), values.data(), 1.0f, destination.columns());
        return;
    }
    if (bias.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> values = bias.bfloat16_values();
        for (size_t row_index = 0; row_index < destination.rows(); ++row_index)
            bfloat16_scaled_add(destination.row(row_index), values.data(), 1.0f, destination.columns());
        return;
    }
    for (size_t row_index = 0; row_index < destination.rows(); ++row_index)
    {
        float* output = destination.row(row_index);
        for (uint32_t column = 0; column < destination.columns(); ++column)
            output[column] += tensor_value(bias, column);
    }
}

void add_batch_inplace(CpuBatch& destination, const CpuBatch& source)
{
    assert(destination.rows() == source.rows() && destination.columns() == source.columns());
    for (size_t row_index = 0; row_index < destination.rows(); ++row_index)
    {
        float* output = destination.row(row_index);
        const float* input = source.row(row_index);
        float_scaled_add(output, input, 1.0f, destination.columns());
    }
}

std::vector<std::vector<float>> batch_to_vectors(const CpuBatch& batch)
{
    std::vector<std::vector<float>> output(batch.rows(), std::vector<float>(batch.columns()));
    for (size_t row_index = 0; row_index < batch.rows(); ++row_index)
        std::copy_n(batch.row(row_index), batch.columns(), output[row_index].begin());
    return output;
}

} // namespace moe
} // namespace ncnn
