#include "backends/ncnn/expertbackend_vulkan.h"
#include "backends/ncnn/gateddeltanet_vulkan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "engine/expertbackend.h"
#include "graph/compiledoperator.h"
#include "kernels/gateddeltanet.h"
#include "kernels/int4.h"
#include "kernels/statecache.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void int4_gpu_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void int4_gpu_check_result(const Result<void>& result, const std::string& message)
{
    if (!result)
        throw std::runtime_error(message + ": " + result.error().message);
}

static TensorData int4_gpu_weight(uint32_t rows, uint32_t columns, uint32_t group_size,
                                  uint32_t seed, bool indexed, bool permuted)
{
    TensorData matrix;
    matrix.dtype = DType::Int4;
    matrix.shape = {rows, columns};
    matrix.quantization_group_size = group_size;
    const size_t stride = columns / 2 + columns % 2;
    const uint32_t groups = columns / group_size + (columns % group_size != 0);
    matrix.quantized_data.assign(static_cast<size_t>(rows) * stride, 0);
    matrix.quantization_scales.resize(static_cast<size_t>(rows) * groups);
    matrix.quantization_zero_points.resize(static_cast<size_t>(rows) * groups);
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            const uint8_t code = static_cast<uint8_t>((row * 11 + column * 7 + column / 3 + seed) % 16);
            matrix.quantized_data[row * stride + column / 2] |= static_cast<uint8_t>(code << ((column & 1u) * 4));
        }
        for (uint32_t group = 0; group < groups; ++group)
        {
            const size_t offset = static_cast<size_t>(row) * groups + group;
            matrix.quantization_scales[offset] = static_cast<float>((row + group + seed) % 3 + 1) / 128.f;
            matrix.quantization_zero_points[offset] = static_cast<uint8_t>((row * 7 + group * 3 + seed) % 17);
        }
    }
    if (indexed)
    {
        matrix.quantization_group_indices.resize(columns);
        for (uint32_t column = 0; column < columns; ++column)
            matrix.quantization_group_indices[column] = (column * 3 + seed) % groups;
    }
    if (permuted)
    {
        matrix.input_column_indices.resize(columns);
        for (uint32_t column = 0; column < columns; ++column)
            matrix.input_column_indices[column] = (column + 1) % columns;
    }
    int4_gpu_check(static_cast<bool>(validate_int4_tensor(matrix)), "valid INT4 Vulkan fixture");
    return matrix;
}

static TensorData int4_gpu_bias(uint32_t columns)
{
    TensorData bias;
    bias.dtype = DType::Float32;
    bias.shape = {columns};
    bias.float32_data.resize(columns);
    for (uint32_t column = 0; column < columns; ++column)
        bias.float32_data[column] = static_cast<float>(static_cast<int>(column % 7) - 3) / 1024.f;
    return bias;
}

static TensorData int4_gpu_bfloat16(uint32_t rows, uint32_t columns)
{
    TensorData matrix;
    matrix.dtype = DType::BFloat16;
    matrix.shape = {rows, columns};
    matrix.bfloat16_data.resize(static_cast<size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            const float value = static_cast<float>(static_cast<int>((row * 5 + column * 3) % 17) - 8) / 128.f;
            matrix.bfloat16_data[static_cast<size_t>(row) * columns + column] = static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16);
        }
    }
    return matrix;
}

static ActivationBuffer int4_gpu_input(size_t rows, uint32_t columns)
{
    ActivationBuffer input(rows, columns);
    for (size_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((row * 7 + column * 13 + column / 5) % 31) - 15) / 64.f;
    return input;
}

static ActivationBuffer int4_gpu_oracle(const TensorData& matrix,
                                        const TensorData* bias,
                                        const ActivationBuffer& input)
{
    ActivationBuffer output(input.rows(), matrix.shape[0]);
    const uint32_t columns = matrix.shape[1];
    const uint32_t groups = matrix.dtype == DType::Int4 ? columns / matrix.quantization_group_size + (columns % matrix.quantization_group_size != 0) : 0;
    const size_t stride = columns / 2 + columns % 2;
    const auto weights = matrix.int4_values();
    for (size_t token = 0; token < input.rows(); ++token)
    {
        for (uint32_t row = 0; row < matrix.shape[0]; ++row)
        {
            double sum = bias ? bias->float32_values()[row] : 0.0;
            for (uint32_t column = 0; column < columns; ++column)
            {
                if (matrix.dtype == DType::BFloat16)
                {
                    const float value = std::bit_cast<float>(static_cast<uint32_t>(matrix.bfloat16_values()[static_cast<size_t>(row) * columns + column]) << 16);
                    sum += static_cast<double>(input.row(token)[column]) * value;
                    continue;
                }
                const uint32_t group = matrix.quantization_group_indices.empty() ? column / matrix.quantization_group_size
                                                                                 : matrix.quantization_group_indices[column];
                const uint32_t logical = matrix.input_column_indices.empty() ? column : matrix.input_column_indices[column];
                const size_t metadata = static_cast<size_t>(row) * groups + group;
                const int code = (weights[row * stride + column / 2] >> ((column & 1u) * 4)) & 15;
                sum += static_cast<double>(input.row(token)[logical])
                       * (code - matrix.quantization_zero_points[metadata]) * matrix.quantization_scales[metadata];
            }
            output.row(token)[row] = static_cast<float>(sum);
        }
    }
    return output;
}

static ActivationBuffer int4_gpu_expert_oracle(const TensorData& gate,
                                               const TensorData* gate_bias,
                                               const TensorData* up,
                                               const TensorData* up_bias,
                                               const TensorData& down,
                                               const TensorData* down_bias,
                                               const ActivationBuffer& input,
                                               ExpertActivation activation = ExpertActivation::Silu,
                                               float activation_limit = 0.f)
{
    const ActivationBuffer first = int4_gpu_oracle(gate, gate_bias, input);
    const ActivationBuffer second = up ? int4_gpu_oracle(*up, up_bias, input) : ActivationBuffer{};
    const uint32_t intermediate = down.shape[1];
    ActivationBuffer activated(input.rows(), intermediate);
    for (size_t token = 0; token < input.rows(); ++token)
    {
        for (uint32_t column = 0; column < intermediate; ++column)
        {
            const uint32_t gate_column = up ? column : gate.qnk_interleave_rows ? column * 2
                                                                                : column;
            const uint32_t up_column = up ? column : gate.qnk_interleave_rows ? column * 2 + 1
                                                                              : intermediate + column;
            float value = first.row(token)[gate_column];
            float linear = up ? second.row(token)[up_column] : first.row(token)[up_column];
            if (activation != ExpertActivation::Silu && activation_limit > 0.f)
            {
                value = std::min(value, activation_limit);
                linear = std::clamp(linear, -activation_limit, activation_limit);
            }
            const float sigmoid_scale = activation == ExpertActivation::GptOssSwiGlu ? 1.702f : 1.f;
            const float up_offset = activation == ExpertActivation::GptOssSwiGlu ? 1.f : 0.f;
            activated.row(token)[column] = value / (1.f + std::exp(-sigmoid_scale * value)) * (linear + up_offset);
        }
    }
    return int4_gpu_oracle(down, down_bias, activated);
}

static void int4_gpu_check_output(const ActivationBuffer& actual, const ActivationBuffer& expected,
                                  const std::string& label)
{
    int4_gpu_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + " output shape");
    float maximum = 0.f;
    for (size_t index = 0; index < expected.values().size(); ++index)
    {
        maximum = std::max(maximum, std::abs(expected.values()[index]));
        const float tolerance = 0.00003f + std::abs(expected.values()[index]) * 0.00003f;
        int4_gpu_check(std::isfinite(actual.values()[index])
                           && std::abs(actual.values()[index] - expected.values()[index]) <= tolerance,
                       label + " scalar oracle at " + std::to_string(index) + ": actual=" + std::to_string(actual.values()[index])
                           + " expected=" + std::to_string(expected.values()[index]));
    }
    int4_gpu_check(maximum > 0.000001f, label + " nonzero scalar oracle");
}

#if NCNN_MOE_WITH_VULKAN
static uint64_t int4_gpu_payload(const TensorData& matrix, uint64_t alignment)
{
    const auto aligned = [alignment](uint64_t bytes) { return (bytes + alignment - 1) / alignment * alignment; };
    const uint64_t raw_size = matrix.dtype == DType::BFloat16 ? matrix.bfloat16_values().size_bytes() : matrix.int4_values().size();
    return aligned(raw_size) + aligned(matrix.shape[0] * sizeof(float))
           + aligned(matrix.quantization_scales.size() * sizeof(float)) + aligned(matrix.quantization_zero_points.size())
           + aligned(matrix.quantization_group_indices.size() * sizeof(uint32_t))
           + aligned(matrix.input_column_indices.size() * sizeof(uint32_t));
}

static void test_int4_gpu_grouped_projection()
{
    const auto runtime = create_vulkan_runtime();
    constexpr uint32_t groups = 3;
    constexpr uint32_t columns = 259;
    constexpr uint32_t outputs = 15;
    const TensorData bias = int4_gpu_bias(outputs);
    for (bool packed : {false, true})
    {
        const TensorData matrix = packed ? int4_gpu_weight(outputs, columns, 64, 7, true, true)
                                         : int4_gpu_bfloat16(outputs, columns);
        auto op = QnkLinear_vulkan::create(matrix, &bias, get_default_gpu_index(), runtime, OptimizationDefaultFlags, groups);
        int4_gpu_check(op && op->input_columns() == columns * groups, "create grouped native projection");
        int4_gpu_check(!QnkLinear_vulkan::create(matrix, &bias, get_default_gpu_index(), runtime, OptimizationDefaultFlags, 4),
                       "reject nondivisible output groups");
        ActivationBuffer input = int4_gpu_input(4, columns * groups);
        ActivationBuffer expected(input.rows(), outputs);
        for (uint32_t group = 0; group < groups; ++group)
        {
            ActivationBuffer grouped(input.rows(), columns);
            for (size_t row = 0; row < input.rows(); ++row)
                std::copy_n(input.row(row) + group * columns, columns, grouped.row(row));
            const ActivationBuffer projected = int4_gpu_oracle(matrix, &bias, grouped);
            for (size_t row = 0; row < input.rows(); ++row)
                std::copy_n(projected.row(row) + group * (outputs / groups), outputs / groups,
                            expected.row(row) + group * (outputs / groups));
        }
        ActivationBuffer actual;
        int4_gpu_check(op->forward(input, actual), "execute grouped native projection");
        int4_gpu_check_output(actual, expected, "grouped projection");
    }
}

static void test_int4_gpu_projections()
{
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(get_default_gpu_index(), runtime, OptimizationDefaultFlags);
    int4_gpu_check(static_cast<bool>(context), "INT4 Vulkan context");
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    for (bool indexed : {false, true})
    {
        for (bool permuted : {false, true})
        {
            for (bool odd_shape : {false, true})
            {
                TensorData matrix = int4_gpu_weight(odd_shape ? 17 : 32, odd_shape ? 33 : 256, odd_shape ? 7 : 128, 3, indexed, permuted);
                TensorData bias = int4_gpu_bias(matrix.shape[0]);
                auto projection = QnkLinear_vulkan::create(matrix, &bias, get_default_gpu_index(), runtime, OptimizationDefaultFlags);
                int4_gpu_check(static_cast<bool>(projection), "create native INT4 Vulkan projection");
                int4_gpu_check(projection->dtype() == DType::Int4 && projection->input_columns() == matrix.shape[1]
                                   && projection->output_columns() == matrix.shape[0],
                               "INT4 Vulkan logical dtype/shape");
                int4_gpu_check(projection->device_bytes() == int4_gpu_payload(matrix, alignment), "INT4 GPU resident bytes are packed payload and aligned metadata");
                if (!odd_shape)
                    int4_gpu_check(projection->device_bytes() < matrix.element_count() * sizeof(uint16_t), "INT4 GPU weights remain smaller than BF16");
                for (size_t rows : {size_t(1), size_t(4), size_t(5)})
                {
                    const ActivationBuffer input = int4_gpu_input(rows, matrix.shape[1]);
                    const ActivationBuffer expected = int4_gpu_oracle(matrix, &bias, input);
                    ActivationBuffer actual;
                    const auto before = get_vulkan_statistics(runtime);
                    int4_gpu_check(projection->forward(input, actual), "execute native INT4 projection");
                    const auto after = get_vulkan_statistics(runtime);
                    int4_gpu_check(after.int4_dispatches > before.int4_dispatches && after.compute_submissions > before.compute_submissions,
                                   "INT4 projection submitted actual Vulkan work");
                    int4_gpu_check_output(actual, expected, "INT4 Vulkan GEMV/GEMM");
                }
                auto invalid = matrix;
                invalid.quantization_scales[0] = -1.f;
                int4_gpu_check(!QnkLinear_vulkan::create(invalid, nullptr, get_default_gpu_index(), runtime, OptimizationDefaultFlags), "reject malformed native INT4 GPU metadata");
            }
        }
    }
}

static void test_int4_gpu_experts()
{
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(get_default_gpu_index(), runtime, OptimizationDefaultFlags);
    int4_gpu_check(static_cast<bool>(context), "INT4 Expert Vulkan context");
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    constexpr uint32_t hidden = 256;
    constexpr uint32_t intermediate = 64;
    auto gate_up = std::make_shared<TensorData>(int4_gpu_weight(intermediate * 2, hidden, 128, 1, true, false));
    auto down = std::make_shared<TensorData>(int4_gpu_weight(hidden, intermediate, 32, 2, false, true));
    auto gate = std::make_shared<TensorData>(int4_gpu_weight(intermediate, hidden, 128, 3, false, true));
    auto up = std::make_shared<TensorData>(int4_gpu_weight(intermediate, hidden, 64, 4, true, false));
    const TensorData gate_up_bias = int4_gpu_bias(intermediate * 2);
    const TensorData gate_bias = int4_gpu_bias(intermediate);
    const TensorData up_bias = int4_gpu_bias(intermediate);
    const TensorData down_bias = int4_gpu_bias(hidden);
    const uint64_t packed_bytes = int4_gpu_payload(*gate_up, alignment) + int4_gpu_payload(*down, alignment);
    const uint64_t separate_bytes = int4_gpu_payload(*gate, alignment) + int4_gpu_payload(*up, alignment) + int4_gpu_payload(*down, alignment);
    for (bool interleaved : {false, true})
    {
        gate_up->qnk_interleave_rows = interleaved;
        auto expert = QnkExpert_vulkan::create(*gate_up, &gate_up_bias, *down, &down_bias, 0.f, get_default_gpu_index(),
                                               ExpertActivation::Silu, runtime, OptimizationDefaultFlags);
        int4_gpu_check(expert && expert->device_bytes() == packed_bytes, "packed INT4 Expert device residency");
        for (size_t rows : {size_t(1), size_t(4), size_t(5)})
        {
            const ActivationBuffer input = int4_gpu_input(rows, hidden);
            const ActivationBuffer expected = int4_gpu_expert_oracle(*gate_up, &gate_up_bias, nullptr, nullptr, *down, &down_bias, input);
            ActivationBuffer actual;
            const auto before = get_vulkan_statistics(runtime);
            int4_gpu_check(expert->forward(input, actual), "execute packed INT4 Expert");
            const auto after = get_vulkan_statistics(runtime);
            int4_gpu_check(after.int4_dispatches == before.int4_dispatches + 2 && after.compute_submissions > before.compute_submissions,
                           "INT4 Expert chain executed quantized Vulkan projections");
            int4_gpu_check_output(actual, expected, "packed INT4 Vulkan Expert");
        }
    }
    auto separate = QnkExpert_vulkan::create_separate(*gate, &gate_bias, *up, &up_bias, *down, &down_bias, 0.f, get_default_gpu_index(),
                                                      ExpertActivation::Silu, runtime, OptimizationDefaultFlags);
    int4_gpu_check(separate && separate->device_bytes() == separate_bytes, "separate INT4 Expert device residency");
    const ActivationBuffer input = int4_gpu_input(4, hidden);
    const ActivationBuffer expected = int4_gpu_expert_oracle(*gate, &gate_bias, up.get(), &up_bias, *down, &down_bias, input);
    ActivationBuffer actual;
    const auto ffn_before = get_vulkan_statistics(runtime);
    int4_gpu_check(separate->forward(input, actual), "execute separate INT4 Expert projections");
    const auto ffn_after = get_vulkan_statistics(runtime);
    int4_gpu_check(ffn_after.compute_submissions == ffn_before.compute_submissions + 1
                       && ffn_after.int4_dispatches == ffn_before.int4_dispatches + 3
                       && ffn_after.batch_uploads == ffn_before.batch_uploads + 1
                       && ffn_after.batch_downloads == ffn_before.batch_downloads + 1,
                   "native dense FFN chain has one submit, three INT4 projections and one host boundary each way");
    int4_gpu_check_output(actual, expected, "separate INT4 Vulkan Expert");

    for (ExpertActivation activation : {ExpertActivation::Silu, ExpertActivation::DeepSeekSwiGlu, ExpertActivation::GptOssSwiGlu})
    {
        constexpr float limit = 0.0625f;
        auto limited = QnkExpert_vulkan::create_separate(*gate, &gate_bias, *up, &up_bias, *down, &down_bias, limit, get_default_gpu_index(),
                                                         activation, runtime, OptimizationDefaultFlags);
        int4_gpu_check(static_cast<bool>(limited), "create limited INT4 Expert activation");
        const auto limited_expected = int4_gpu_expert_oracle(*gate, &gate_bias, up.get(), &up_bias, *down, &down_bias, input, activation, limit);
        int4_gpu_check(limited->forward(input, actual), "execute limited INT4 Expert activation");
        int4_gpu_check_output(actual, limited_expected, "Silu ignores limit; DeepSeek/GPT-OSS retain clamp");
        if (activation == ExpertActivation::Silu)
        {
            const auto clamped = int4_gpu_expert_oracle(*gate, &gate_bias, up.get(), &up_bias, *down, &down_bias, input, ExpertActivation::DeepSeekSwiGlu, limit);
            float difference = 0.f;
            for (size_t index = 0; index < clamped.values().size(); ++index)
                difference = std::max(difference, std::abs(clamped.values()[index] - limited_expected.values()[index]));
            int4_gpu_check(difference > 0.001f, "Silu limit fixture distinguishes an incorrect clamp");
        }
    }

    const TensorData bfloat_up = int4_gpu_bfloat16(intermediate, hidden);
    const uint64_t mixed_bytes = int4_gpu_payload(*gate, alignment) + int4_gpu_payload(bfloat_up, alignment) + int4_gpu_payload(*down, alignment);
    auto mixed = QnkExpert_vulkan::create_separate(*gate, &gate_bias, bfloat_up, &up_bias, *down, &down_bias, 0.f, get_default_gpu_index(),
                                                   ExpertActivation::Silu, runtime, OptimizationDefaultFlags);
    int4_gpu_check(mixed && mixed->device_bytes() == mixed_bytes,
                   "mixed INT4/BF16 Expert retains each original device storage type");
    const auto mixed_before = get_vulkan_statistics(runtime);
    int4_gpu_check(mixed->forward(input, actual), "execute mixed INT4/BF16 Vulkan Expert");
    const auto mixed_after = get_vulkan_statistics(runtime);
    int4_gpu_check(mixed_after.int4_dispatches == mixed_before.int4_dispatches + 2, "mixed Expert executes INT4 projections on GPU");
    int4_gpu_check_output(actual, int4_gpu_expert_oracle(*gate, &gate_bias, &bfloat_up, &up_bias, *down, &down_bias, input), "mixed INT4/BF16 Vulkan Expert");

    gate_up->qnk_interleave_rows = false;
    auto backend = create_vulkan_expert_backend(packed_bytes + separate_bytes + mixed_bytes, get_default_gpu_index(), nullptr, runtime, OptimizationDefaultFlags);
    int4_gpu_check(static_cast<bool>(backend), "INT4 resident Expert backend");
    std::shared_ptr<const void> packed_pin;
    std::shared_ptr<const void> separate_pin;
    std::shared_ptr<const void> mixed_pin;
    int4_gpu_check(backend->prepare_demand("int4-packed", gate_up, &gate_up_bias, down, &down_bias, 0, 0.f, ExpertActivation::Silu, packed_pin), "admit packed INT4 Expert on GPU");
    int4_gpu_check(backend->prepare_demand_separate("int4-separate", gate, &gate_bias, up, &up_bias, down, &down_bias,
                                                    0, 0.f, ExpertActivation::Silu, separate_pin),
                   "admit separate INT4 Expert on GPU");
    int4_gpu_check(backend->prepare_demand_separate("int4-mixed", gate, &gate_bias, std::make_shared<TensorData>(bfloat_up), &up_bias, down, &down_bias,
                                                    0, 0.f, ExpertActivation::Silu, mixed_pin),
                   "admit mixed INT4/BF16 Expert on GPU");
    const auto admitted = backend->statistics();
    int4_gpu_check(packed_pin && separate_pin && mixed_pin && admitted.stores == 3 && admitted.resident_size == packed_bytes + separate_bytes + mixed_bytes,
                   "Expert cache accounts exact low-bit GPU resident bytes");
    std::array<ActivationBuffer, 2> outputs = {ActivationBuffer(1, 1), ActivationBuffer(1, 1)};
    for (auto& output : outputs)
        output.row(0)[0] = 123.f;
    const std::array<ExpertBackendRequest, 2> requests = {
        ExpertBackendRequest{"int4-packed", &input, &outputs[0]},
        ExpertBackendRequest{"int4-separate", &input, &outputs[1]},
    };
    const auto before = get_vulkan_statistics(runtime);
    auto submission = backend->submit_batch(requests);
    int4_gpu_check(static_cast<bool>(submission), "submit native INT4 Expert batch");
    for (const auto reservation : submission->reservations())
        int4_gpu_check(reservation == ExpertBackendExecutionResult::Executed, "INT4 Expert GPU reservation");
    const auto results = submission->wait();
    int4_gpu_check(results.size() == 2, "INT4 Expert GPU result count");
    for (const auto result : results)
        int4_gpu_check(result == ExpertBackendExecutionResult::Executed, "INT4 Expert actually executed on GPU");
    for (const auto& output : outputs)
        int4_gpu_check(output.rows() == 1 && output.columns() == 1 && output.row(0)[0] == 123.f, "Expert result stays private before commit");
    int4_gpu_check(submission->commit(), "commit native INT4 Expert GPU batch");
    const auto after = get_vulkan_statistics(runtime);
    int4_gpu_check(after.int4_dispatches == before.int4_dispatches + 5
                       && backend->statistics().executions >= admitted.executions + 2,
                   "native INT4 batch publishes GPU execution counters");
    int4_gpu_check_output(outputs[0], int4_gpu_expert_oracle(*gate_up, &gate_up_bias, nullptr, nullptr, *down, &down_bias, input), "resident packed INT4 Expert");
    int4_gpu_check_output(outputs[1], expected, "resident separate INT4 Expert");
    const std::array<ExpertBackendRequest, 2> mixed_requests = {
        ExpertBackendRequest{"int4-separate", &input, &outputs[0]},
        ExpertBackendRequest{"int4-mixed", &input, &outputs[1]},
    };
    const auto batch_before = get_vulkan_statistics(runtime);
    auto batch = backend->submit_batch(mixed_requests);
    int4_gpu_check(static_cast<bool>(batch), "submit mixed INT4/BF16 Expert batch");
    const auto batch_results = batch->wait();
    int4_gpu_check(batch_results.size() == 2 && std::all_of(batch_results.begin(), batch_results.end(), [](auto result) { return result == ExpertBackendExecutionResult::Executed; }),
                   "mixed batch executes both Experts on GPU");
    int4_gpu_check(batch->commit(), "commit mixed INT4/BF16 Expert batch");
    const auto batch_after = get_vulkan_statistics(runtime);
    int4_gpu_check(batch_after.int4_dispatches == batch_before.int4_dispatches + 5
                       && batch_after.dispatches == batch_before.dispatches + 8
                       && batch_after.compute_submissions == batch_before.compute_submissions + 1,
                   "batched Experts publish exact totals for INT4 and BF16 projections");
    int4_gpu_check_output(outputs[0], expected, "batched INT4 Expert");
    int4_gpu_check_output(outputs[1], int4_gpu_expert_oracle(*gate, &gate_bias, &bfloat_up, &up_bias, *down, &down_bias, input), "batched mixed Expert");

    auto insufficient = create_vulkan_expert_backend(packed_bytes - 1, get_default_gpu_index(), nullptr, runtime, OptimizationDefaultFlags);
    std::shared_ptr<const void> rejected;
    int4_gpu_check(!insufficient->prepare_demand("too-large", gate_up, &gate_up_bias, down, &down_bias, 0, 0.f,
                                                 ExpertActivation::Silu, rejected)
                       && !rejected,
                   "GPU admission budget includes packed metadata and alignment");
}

static TensorData int4_gpu_float_reference(const TensorData& matrix)
{
    TensorData result;
    result.dtype = DType::Float32;
    result.shape = matrix.shape;
    result.float32_data.resize(static_cast<size_t>(matrix.element_count()));
    const uint32_t columns = matrix.shape[1];
    const uint32_t groups = columns / matrix.quantization_group_size + (columns % matrix.quantization_group_size != 0);
    const size_t stride = columns / 2 + columns % 2;
    for (uint32_t row = 0; row < matrix.shape[0]; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            const uint32_t group = matrix.quantization_group_indices.empty() ? column / matrix.quantization_group_size
                                                                             : matrix.quantization_group_indices[column];
            const uint32_t logical = matrix.input_column_indices.empty() ? column : matrix.input_column_indices[column];
            const size_t metadata = static_cast<size_t>(row) * groups + group;
            const int code = (matrix.int4_values()[row * stride + column / 2] >> ((column & 1u) * 4)) & 15;
            result.float32_data[static_cast<size_t>(row) * columns + logical] = (code - matrix.quantization_zero_points[metadata]) * matrix.quantization_scales[metadata];
        }
    }
    return result;
}

static TensorData int4_gpu_float_tensor(std::vector<uint32_t> shape, float value)
{
    TensorData result;
    result.dtype = DType::Float32;
    result.shape = std::move(shape);
    result.float32_data.assign(static_cast<size_t>(result.element_count()), value);
    return result;
}

static void test_int4_gpu_gated_delta()
{
    constexpr uint32_t hidden = 16;
    constexpr uint32_t heads = 3;
    constexpr uint32_t keys = 1;
    constexpr uint32_t key_dimension = 4;
    constexpr uint32_t value_dimension = 4;
    constexpr uint32_t convolution_size = 2 * keys * key_dimension + heads * value_dimension;
    const auto runtime = create_vulkan_runtime();
    const std::array<TensorData, 4> inputs = {
        int4_gpu_weight(convolution_size, hidden, 5, 1, true, false),
        int4_gpu_weight(heads * value_dimension, hidden, 5, 2, false, true),
        int4_gpu_weight(heads, hidden, 5, 3, true, false),
        int4_gpu_weight(heads, hidden, 5, 4, false, false),
    };
    const TensorData output = int4_gpu_weight(hidden, heads * value_dimension, 5, 5, true, true);
    int4_gpu_check(output.input_column_indices[output.input_column_indices[0]] != 0,
                   "DeltaNet output map is not its own inverse");
    const TensorData convolution = int4_gpu_float_tensor({convolution_size, 1, 2}, 0.25f);
    const TensorData time_bias = int4_gpu_float_tensor({heads}, -0.25f);
    const TensorData decay_log = int4_gpu_float_tensor({heads}, -0.5f);
    const TensorData norm = int4_gpu_float_tensor({value_dimension}, 1.f);
    std::array<std::shared_ptr<QnkLinear_vulkan>, 4> projections;
    for (size_t index = 0; index < inputs.size(); ++index)
    {
        projections[index] = QnkLinear_vulkan::create(inputs[index], nullptr, get_default_gpu_index(), runtime, OptimizationDefaultFlags);
        int4_gpu_check(static_cast<bool>(projections[index]), "native DeltaNet input projection");
    }
    auto output_projection = QnkLinear_vulkan::create(output, nullptr, get_default_gpu_index(), runtime, OptimizationDefaultFlags);
    auto operation = GatedDeltaNet_vulkan::create_quantized(projections, convolution, time_bias, decay_log, norm,
                                                            output_projection, heads, keys, key_dimension, value_dimension,
                                                            2, 1e-6f, false, get_default_gpu_index(), runtime, OptimizationDefaultFlags);
    int4_gpu_check(static_cast<bool>(operation), "create native INT4 DeltaNet recurrence");

    WeightStore weights;
    const auto add = [&](const char* name, TensorData tensor) {
        auto handle = weights.add(name, std::move(tensor));
        int4_gpu_check(static_cast<bool>(handle), "add independent float DeltaNet reference");
        return handle.value();
    };
    AttentionBlockPlan plan;
    plan.kind = AttentionKind::GatedDeltaNet;
    plan.head_count = heads;
    plan.kv_head_count = keys;
    plan.head_dimension = key_dimension;
    plan.value_head_dimension = value_dimension;
    plan.convolution_kernel_size = 2;
    plan.norm_epsilon = 1e-6f;
    plan.flags = AttentionBlockExternalResidual;
    plan.delta_qkv_weight = add("qkv", int4_gpu_float_reference(inputs[0]));
    plan.delta_z_weight = add("z", int4_gpu_float_reference(inputs[1]));
    plan.delta_beta_weight = add("beta", int4_gpu_float_reference(inputs[2]));
    plan.delta_alpha_weight = add("alpha", int4_gpu_float_reference(inputs[3]));
    plan.output_weight = add("output", int4_gpu_float_reference(output));
    plan.delta_convolution_weight = add("convolution", convolution);
    plan.delta_time_bias = add("time", time_bias);
    plan.delta_decay_log = add("decay", decay_log);
    plan.delta_norm_weight = add("norm", norm);
    CompiledOperatorTable operators;
    operators.bind_weight_count(weights.size());
    plan.gated_delta_vulkan_operator = operators.allocate();
    operators.at_mutable(plan.gated_delta_vulkan_operator).gated_delta = operation;
    std::array<LayerCache, 1> cpu_cache;
    std::array<LayerCache, 1> gpu_cache;
    GatedDeltaScratch cpu_scratch;
    GatedDeltaScratch gpu_scratch;
    const auto execute = [&](size_t rows) {
        const ActivationBuffer input = int4_gpu_input(rows, hidden);
        ActivationBuffer expected;
        ActivationBuffer actual;
        int4_gpu_check_result(forward_gated_delta(weights, operators, plan, ExecutionBackend::Cpu,
                                                  cpu_cache[0], cpu_scratch, input, expected, OptimizationDefaultFlags),
                              "independent CPU DeltaNet reference");
        const auto before = get_vulkan_statistics(runtime);
        const bool existing_state = static_cast<bool>(gpu_cache[0].gated_delta_device_state);
        int4_gpu_check_result(forward_gated_delta(weights, operators, plan, ExecutionBackend::Vulkan,
                                                  gpu_cache[0], gpu_scratch, input, actual, OptimizationDefaultFlags),
                              "native GPU DeltaNet continuation (rows=" + std::to_string(rows)
                                  + ", transaction=" + (gpu_cache[0].transaction.active ? "active" : "inactive") + ")");
        const auto after = get_vulkan_statistics(runtime);
        int4_gpu_check(after.gated_delta_submissions == before.gated_delta_submissions + 1
                           && after.int4_dispatches == before.int4_dispatches + 5,
                       "DeltaNet recurrence and all five native projections executed on GPU");
        if (existing_state)
            int4_gpu_check(after.compute_submissions == before.compute_submissions + 1, "native DeltaNet chain uses one GPU submit");
        int4_gpu_check_output(actual, expected, "native INT4 DeltaNet stateful output");
    };
    const auto compare_state = [&] {
        int4_gpu_check(gpu_cache[0].gated_delta_device_state && gpu_cache[0].gated_delta_token_count == cpu_cache[0].gated_delta_token_count,
                       "DeltaNet GPU state and token count remain authoritative");
        std::vector<float> gpu_convolution;
        std::vector<float> gpu_recurrent;
        int4_gpu_check(gpu_cache[0].gated_delta_device_state->download(gpu_convolution, gpu_recurrent), "read native DeltaNet state oracle");
        const auto compare = [](const std::vector<float>& actual, const std::vector<float>& expected) {
            int4_gpu_check(actual.size() == expected.size(), "DeltaNet state vector shape");
            for (size_t index = 0; index < actual.size(); ++index)
                int4_gpu_check(std::abs(actual[index] - expected[index]) < 0.00003f + std::abs(expected[index]) * 0.00003f,
                               "DeltaNet GPU recurrent/convolution state equals independent CPU reference");
        };
        compare(gpu_convolution, cpu_cache[0].gated_delta_convolution);
        compare(gpu_recurrent, cpu_cache[0].gated_delta_recurrent);
    };
    // A fresh multi-row prefill must initialize authoritative device state.
    execute(3);
    compare_state();
    execute(1);
    compare_state();
    const auto* persistent = gpu_cache[0].gated_delta_device_state.get();
    execute(3);
    compare_state();
    // Device transactions retain one state snapshot per decode call, as in
    // the existing BF16 path; multi-token continuation above is nontransactional.
    int4_gpu_check_result(begin_state_cache_transaction(cpu_cache, 3), "begin CPU DeltaNet transaction");
    int4_gpu_check_result(begin_state_cache_transaction(gpu_cache, 3), "begin native DeltaNet transaction");
    for (size_t row = 0; row < 3; ++row)
        execute(1);
    int4_gpu_check_result(finish_state_cache_transaction(cpu_cache, 1), "partial commit CPU DeltaNet transaction");
    int4_gpu_check_result(finish_state_cache_transaction(gpu_cache, 1), "partial commit native DeltaNet transaction");
    compare_state();
    execute(1);
    compare_state();
    int4_gpu_check_result(begin_state_cache_transaction(cpu_cache, 2), "begin CPU DeltaNet rollback");
    int4_gpu_check_result(begin_state_cache_transaction(gpu_cache, 2), "begin native DeltaNet rollback");
    for (size_t row = 0; row < 2; ++row)
        execute(1);
    int4_gpu_check_result(finish_state_cache_transaction(cpu_cache, 0), "rollback CPU DeltaNet transaction");
    int4_gpu_check_result(finish_state_cache_transaction(gpu_cache, 0), "rollback native DeltaNet transaction");
    compare_state();
    execute(1);
    compare_state();
    int4_gpu_check(gpu_cache[0].gated_delta_device_state.get() == persistent, "native DeltaNet preserves its device state allocation");

    // CPU state may precede GPU admission; the next multi-row call migrates it
    // once and consumes every continuation row in the same device chain.
    LayerCache migration_reference;
    LayerCache migration_cache;
    GatedDeltaScratch migration_scratch;
    GatedDeltaScratch migration_gpu_scratch;
    ActivationBuffer migration_expected;
    ActivationBuffer migration_actual;
    const ActivationBuffer migration_prefix = int4_gpu_input(2, hidden);
    int4_gpu_check_result(forward_gated_delta(weights, operators, plan, ExecutionBackend::Cpu,
                                              migration_reference, migration_scratch, migration_prefix,
                                              migration_expected, OptimizationDefaultFlags),
                          "CPU migration prefix");
    migration_cache = migration_reference;
    const ActivationBuffer migration_input = int4_gpu_input(3, hidden);
    int4_gpu_check_result(forward_gated_delta(weights, operators, plan, ExecutionBackend::Cpu,
                                              migration_reference, migration_scratch, migration_input,
                                              migration_expected, OptimizationDefaultFlags),
                          "CPU migration continuation");
    int4_gpu_check_result(forward_gated_delta(weights, operators, plan, ExecutionBackend::Vulkan,
                                              migration_cache, migration_gpu_scratch, migration_input,
                                              migration_actual, OptimizationDefaultFlags),
                          "GPU multi-row CPU-state migration");
    int4_gpu_check_output(migration_actual, migration_expected, "migrated native DeltaNet continuation");
    int4_gpu_check(migration_cache.gated_delta_token_count == 5 && migration_cache.gated_delta_device_state
                       && migration_cache.gated_delta_convolution.empty() && migration_cache.gated_delta_recurrent.empty(),
                   "GPU migration commits logical rows and releases CPU state");

    const auto& context = operation->vulkan_context();
    const auto& option = operation->option();
    const ActivationBuffer recorded_input = int4_gpu_input(3, hidden);
    LayerCache recorded_reference;
    LayerCache recorded_cache;
    GatedDeltaScratch recorded_reference_scratch;
    ActivationBuffer recorded_expected;
    ActivationBuffer recorded_actual(3, hidden);
    int4_gpu_check_result(forward_gated_delta(weights, operators, plan, ExecutionBackend::Cpu,
                                              recorded_reference, recorded_reference_scratch, recorded_input,
                                              recorded_expected, OptimizationDefaultFlags),
                          "recorded CPU prefill oracle");
    const auto record_before = get_vulkan_statistics(runtime);
    {
        GatedDeltaWork_vulkan work;
        auto lease = context->acquire_transfer_slot();
        auto& slot = lease.slot();
        int4_gpu_check(fill_staging_upload(recorded_input, slot.upload, slot.staging_allocator)
                           && prepare_staging_batch(slot.download, 3, hidden, slot.staging_allocator),
                       "record API staging fixture");
        std::unique_lock<std::mutex> lock(context->command_mutex());
        auto& command = *slot.command;
        int4_gpu_check(!slot.command_used || command.reset() == 0, "reset record API command");
        slot.command_used = true;
        ncnn::VkMat device_input;
        ncnn::VkMat device_output;
        int4_gpu_check(record_mapped_upload(slot.upload, device_input, command, option), "record API input upload");
        int4_gpu_check(operation->record_forward(device_input, recorded_cache, 3, device_output, command, work), "fresh device-only GDN recording");
        const auto record_pending = get_vulkan_statistics(runtime);
        int4_gpu_check(recorded_cache.gated_delta_token_count == 0
                           && record_pending.gated_delta_fusions == record_before.gated_delta_fusions
                           && record_pending.compute_submissions == record_before.compute_submissions,
                       "recording does not publish token counts or GPU completion counters");
        GatedDeltaWork_vulkan duplicate;
        ncnn::VkMat duplicate_output;
        int4_gpu_check(!operation->record_forward(device_input, recorded_cache, 3, duplicate_output, command, duplicate),
                       "overlapping recordings of one recurrent state are rejected");
        int4_gpu_check(record_prepared_activation_staging_download(device_output, 3, hidden, slot.download,
                                                                   command, context->device(), option, DType::Float32),
                       "record API output oracle readback");
        int4_gpu_check(submit_compute_and_wait(command, context->device()) == 0, "submit recorded GDN prefill");
        int4_gpu_check(work.commit() && !work.commit(), "GDN work commits exactly once");
        int4_gpu_check(copy_staging_to_cpu_batch(slot.download, recorded_actual), "materialize GDN test oracle only");
    }
    int4_gpu_check_output(recorded_actual, recorded_expected, "recorded fresh prefill output");
    const auto record_after = get_vulkan_statistics(runtime);
    int4_gpu_check(recorded_cache.gated_delta_token_count == 3
                       && record_after.gated_delta_fusions == record_before.gated_delta_fusions + 1
                       && record_after.compute_submissions == record_before.compute_submissions
                       && record_after.batch_uploads == record_before.batch_uploads
                       && record_after.batch_downloads == record_before.batch_downloads,
                   "record API leaves submission and activation transfer accounting to its caller");

    LayerCache abandoned_cache;
    {
        GatedDeltaWork_vulkan work;
        auto lease = context->acquire_transfer_slot();
        auto& slot = lease.slot();
        int4_gpu_check(fill_staging_upload(recorded_input, slot.upload, slot.staging_allocator), "abandoned record input");
        std::unique_lock<std::mutex> lock(context->command_mutex());
        auto& command = *slot.command;
        int4_gpu_check(!slot.command_used || command.reset() == 0, "reset abandoned record command");
        slot.command_used = true;
        ncnn::VkMat device_input;
        ncnn::VkMat device_output;
        int4_gpu_check(record_mapped_upload(slot.upload, device_input, command, option)
                           && operation->record_forward(device_input, abandoned_cache, 3, device_output, command, work),
                       "record disposable fresh state");
        int4_gpu_check(command.reset() == 0, "discard unsubmitted command before work storage");
    }
    int4_gpu_check(!abandoned_cache.gated_delta_device_state && abandoned_cache.gated_delta_token_count == 0
                       && abandoned_cache.device_allocated_size == 0,
                   "abandoned recording restores fresh cache ownership and logical metadata");

    // A failed caller submit must poison existing authoritative state, rather
    // than silently download it and execute a CPU approximation.
    {
        GatedDeltaWork_vulkan work;
        auto lease = context->acquire_transfer_slot();
        auto& slot = lease.slot();
        int4_gpu_check(fill_staging_upload(recorded_input, slot.upload, slot.staging_allocator), "failed-submit record input");
        std::unique_lock<std::mutex> lock(context->command_mutex());
        auto& command = *slot.command;
        int4_gpu_check(!slot.command_used || command.reset() == 0, "reset failed-submit record command");
        slot.command_used = true;
        ncnn::VkMat device_input;
        ncnn::VkMat device_output;
        int4_gpu_check(record_mapped_upload(slot.upload, device_input, command, option)
                           && operation->record_forward(device_input, recorded_cache, 3, device_output, command, work),
                       "record existing authoritative state");
        work.mark_submit_failed();
        int4_gpu_check(!work.commit() && command.reset() == 0, "failed work cannot publish a completion");
    }
    GatedDeltaScratch failed_scratch;
    ActivationBuffer failed_output;
    const auto failed = forward_gated_delta(weights, operators, plan, ExecutionBackend::Vulkan,
                                            recorded_cache, failed_scratch, recorded_input, failed_output, OptimizationDefaultFlags);
    int4_gpu_check(!failed && failed.error().code == ErrorCode::InternalError
                       && recorded_cache.gated_delta_token_count == 3 && recorded_cache.gated_delta_device_state
                       && recorded_cache.gated_delta_convolution.empty() && recorded_cache.gated_delta_recurrent.empty(),
                   "failed Vulkan state never executes a silent CPU fallback");
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
        ncnn::moe::test_int4_gpu_projections();
        ncnn::moe::test_int4_gpu_grouped_projection();
        ncnn::moe::test_int4_gpu_experts();
        ncnn::moe::test_int4_gpu_gated_delta();
        std::cout << "Native INT4 Vulkan projection and Expert tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Native INT4 Vulkan test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
