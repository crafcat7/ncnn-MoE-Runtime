#include "backends/ncnn/expertbackend_vulkan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/moecombine_vulkan.h"
#include "backends/ncnn/vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "engine/expertbackend.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static void iq4_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static TensorData block_matrix(DType dtype, uint32_t rows, uint32_t columns, uint32_t seed, bool extreme = false)
{
    TensorData matrix;
    matrix.dtype = dtype;
    matrix.shape = {rows, columns};
    const size_t block_bytes = dtype == DType::IQ4NL ? 18 : dtype == DType::Q5_1 ? 24
                                                        : dtype == DType::Q4K    ? 144
                                                                                 : 34;
    const size_t size = static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns));
    iq4_check(size == static_cast<size_t>(rows) * (columns / qnk_block_size(dtype)) * block_bytes, "native GGUF block byte count");
    // Unaligned host mapping, odd block counts and alternating native row
    // strides exercise byte/half reads without widening the source weights.
    std::shared_ptr<uint8_t[]> owner(new uint8_t[size + 1]);
    uint8_t* bytes = owner.get() + 1;
    constexpr std::array<uint16_t, 6> scales = {0x1800, 0x9800, 0x2000, 0x0001, 0x8001, 0x0000};
    for (size_t block = 0; block < size / block_bytes; ++block)
    {
        uint8_t* encoded = bytes + block * block_bytes;
        std::fill_n(encoded, block_bytes, uint8_t(0));
        const uint16_t scale = extreme ? (block % 2 == 0 ? 0x7bff : 0xfbff) : scales[(block + seed) % scales.size()];
        encoded[0] = static_cast<uint8_t>(scale);
        encoded[1] = static_cast<uint8_t>(scale >> 8);
        if (dtype == DType::Q4K)
        {
            // All eight Q4_K local scales are exactly one, dmin is zero.
            std::fill_n(encoded + 4, 12, uint8_t(1));
            for (uint32_t lane = 0; lane < 128; ++lane)
                encoded[16 + lane] = static_cast<uint8_t>(((lane + block + seed) & 15)
                                                          | (((lane * 3 + block + seed + 1) & 15) << 4));
        }
        else if (dtype == DType::IQ4NL)
        {
            for (uint32_t lane = 0; lane < 16; ++lane)
            {
                const uint8_t low = static_cast<uint8_t>((lane * 5 + block + seed) & 15);
                const uint8_t high = static_cast<uint8_t>((lane * 3 + block * 7 + seed + 1) & 15);
                encoded[2 + lane] = static_cast<uint8_t>(low | (high << 4));
            }
        }
        else if (dtype == DType::Q5_1)
        {
            const uint16_t minimum = block % 3 == 0 ? 0x9400 : block % 3 == 1 ? 0x1400
                                                                              : 0x8001;
            encoded[2] = static_cast<uint8_t>(minimum);
            encoded[3] = static_cast<uint8_t>(minimum >> 8);
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                // Every block covers all 32 codes and both halves of qh.
                const uint8_t code = static_cast<uint8_t>((lane * 5 + block + seed) & 31);
                encoded[8 + lane % 16] |= static_cast<uint8_t>((code & 15) << (lane < 16 ? 0 : 4));
                encoded[4 + lane / 8] |= static_cast<uint8_t>((code >> 4) << (lane % 8));
            }
        }
        else
        {
            for (uint32_t lane = 0; lane < 32; ++lane)
                encoded[2 + lane] = static_cast<uint8_t>(static_cast<int>((lane * 17 + block + seed) % 256) - 128);
        }
    }
    matrix.mapped_data = std::shared_ptr<const uint8_t>(owner, bytes);
    matrix.mapped_size = size;
    return matrix;
}

static ActivationBuffer iq4_input(size_t rows, uint32_t columns)
{
    ActivationBuffer input(rows, columns);
    for (size_t token = 0; token < rows; ++token)
        for (uint32_t column = 0; column < columns; ++column)
            input.row(token)[column] = static_cast<float>(static_cast<int>((token * 13 + column * 7 + column / 5) % 41) - 20) / 32.f;
    return input;
}

// Independent GGML-format oracle: no runtime dequantizer, half converter or
// qnk dot routine participates in the expected result.
static float iq4_half(uint8_t low, uint8_t high)
{
    const uint32_t bits = static_cast<uint32_t>(low) | (static_cast<uint32_t>(high) << 8);
    const uint32_t exponent = (bits >> 10) & 31;
    const uint32_t fraction = bits & 1023;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24)
                                          : std::ldexp(static_cast<float>(1024 + fraction), static_cast<int>(exponent) - 25);
    return bits & 32768 ? -magnitude : magnitude;
}

static ActivationBuffer iq4_reference(const TensorData& matrix, const ActivationBuffer& input, uint32_t groups = 1)
{
    constexpr std::array<int, 16> codebook = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    ActivationBuffer output(input.rows(), matrix.shape[0]);
    const uint32_t columns = matrix.shape[1];
    const auto raw = matrix.qnk_values();
    for (size_t token = 0; token < input.rows(); ++token)
        for (uint32_t row = 0; row < matrix.shape[0]; ++row)
        {
            const uint32_t group = row / (matrix.shape[0] / groups);
            double sum = 0;
            for (uint32_t column = 0; column < columns; ++column)
            {
                const size_t block_bytes = matrix.dtype == DType::IQ4NL ? 18 : matrix.dtype == DType::Q5_1 ? 24
                                                                           : matrix.dtype == DType::Q4K    ? 144
                                                                                                           : 34;
                const uint32_t block_columns = matrix.dtype == DType::Q4K ? 256 : 32;
                const size_t offset = (static_cast<size_t>(row) * (columns / block_columns) + column / block_columns) * block_bytes;
                const uint32_t lane = column % block_columns;
                const float scale = iq4_half(raw[offset], raw[offset + 1]);
                float value;
                if (matrix.dtype == DType::Q4K)
                {
                    const uint8_t packed = raw[offset + 16 + lane / 64 * 32 + lane % 32];
                    value = scale * (lane % 64 < 32 ? packed & 15 : packed >> 4);
                }
                else if (matrix.dtype == DType::IQ4NL)
                {
                    const uint8_t packed = raw[offset + 2 + lane % 16];
                    const uint32_t code = lane < 16 ? packed & 15 : packed >> 4;
                    value = scale * codebook[code];
                }
                else if (matrix.dtype == DType::Q5_1)
                {
                    const uint32_t high_bits = static_cast<uint32_t>(raw[offset + 4])
                                               | (static_cast<uint32_t>(raw[offset + 5]) << 8)
                                               | (static_cast<uint32_t>(raw[offset + 6]) << 16)
                                               | (static_cast<uint32_t>(raw[offset + 7]) << 24);
                    const uint8_t packed = raw[offset + 8 + lane % 16];
                    const uint32_t low = lane < 16 ? packed & 15 : packed >> 4;
                    const uint32_t code = low | (((high_bits >> lane) & 1) << 4);
                    value = scale * code + iq4_half(raw[offset + 2], raw[offset + 3]);
                }
                else
                {
                    const int byte = raw[offset + 2 + lane];
                    value = scale * (byte < 128 ? byte : byte - 256);
                }
                const uint32_t logical = matrix.input_column_indices.empty() ? column : matrix.input_column_indices[column];
                sum += static_cast<double>(value) * input.row(token)[group * columns + logical];
            }
            output.row(token)[row] = static_cast<float>(sum);
        }
    return output;
}

static void iq4_compare(const ActivationBuffer& actual, const ActivationBuffer& expected, const std::string& label)
{
    iq4_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + ": shape");
    for (size_t index = 0; index < expected.values().size(); ++index)
    {
        const float reference = expected.values()[index];
        iq4_check(std::isfinite(actual.values()[index])
                      && std::abs(actual.values()[index] - reference) <= 0.00002f + 0.00003f * std::abs(reference),
                  label + ": element " + std::to_string(index) + " actual=" + std::to_string(actual.values()[index])
                      + " expected=" + std::to_string(reference));
    }
}

static void iq4_projections(uint32_t device, DType dtype)
{
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    iq4_check(static_cast<bool>(context), "IQ4_NL/Q5_1 projection context");
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    const auto align = [alignment](uint64_t value) { return (value + alignment - 1) / alignment * alignment; };
    for (uint32_t columns : {32u, 64u, 96u, 288u})
        for (uint32_t groups : {1u, 2u})
        {
            const TensorData matrix = block_matrix(dtype, groups == 1 ? 5 : 6, columns, columns);
            auto operation = QnkLinear_vulkan::create(matrix, nullptr, device, runtime, OptimizationDefaultFlags, groups);
            const uint64_t expected_storage = align(matrix.qnk_values().size()) + align(matrix.shape[0] * sizeof(float));
            iq4_check(operation && operation->dtype() == dtype
                          && operation->device_bytes() == expected_storage
                          && QnkLinear_vulkan::storage_bytes(matrix, alignment) == expected_storage,
                      "IQ4_NL/Q5_1 GPU storage contains native bytes and bias only");
            for (size_t tokens : {size_t(1), size_t(4), size_t(5)})
            {
                const auto input = iq4_input(tokens, columns * groups);
                const auto expected = iq4_reference(matrix, input, groups);
                ActivationBuffer actual;
                const auto before = get_vulkan_statistics(runtime);
                iq4_check(operation->forward(input, actual), "IQ4_NL/Q5_1 native projection dispatch");
                iq4_check(get_vulkan_statistics(runtime).dispatches == before.dispatches + 1,
                          "IQ4_NL/Q5_1 performs one native projection");
                iq4_compare(actual, expected, "IQ4_NL/Q5_1 grouped/misaligned/block-tail reference");
                if (groups == 1)
                    iq4_compare(forward_linear(matrix, input, OptimizationDefaultFlags), expected, "IQ4_NL/Q5_1 CPU native reference");
            }
        }
    for (bool extreme : {false, true})
    {
        TensorData mapped = block_matrix(dtype, 5, 96, 7, extreme);
        mapped.input_column_indices.resize(96);
        for (uint32_t column = 0; column < 96; ++column)
            mapped.input_column_indices[column] = 95 - column;
        auto operation = QnkLinear_vulkan::create(mapped, nullptr, device, runtime, OptimizationDefaultFlags);
        const auto input = iq4_input(4, 96);
        ActivationBuffer actual;
        iq4_check(operation && operation->forward(input, actual), "IQ4_NL/Q5_1 mapped-column projection");
        iq4_compare(actual, iq4_reference(mapped, input), "IQ4_NL/Q5_1 column map and finite extreme scales");
    }
    TensorData invalid = block_matrix(dtype, 5, 32, 1);
    invalid.shape[1] = 33;
    iq4_check(!QnkLinear_vulkan::create(invalid, nullptr, device, runtime, OptimizationDefaultFlags), "reject incomplete IQ4_NL/Q5_1 block");
    invalid = block_matrix(dtype, 5, 32, 1);
    --invalid.mapped_size;
    iq4_check(!QnkLinear_vulkan::create(invalid, nullptr, device, runtime, OptimizationDefaultFlags), "reject truncated IQ4_NL/Q5_1 bytes");
    const TensorData vocabulary = block_matrix(dtype, 65537, 32, 1);
    auto head = QnkLinear_vulkan::create(vocabulary, nullptr, device, runtime, OptimizationDefaultFlags);
    const auto input = iq4_input(1, 32);
    ActivationBuffer logits;
    iq4_check(head && head->forward(input, logits), "IQ4_NL/Q5_1 LM Head exceeds 65535 workgroups");
    iq4_compare(logits, iq4_reference(vocabulary, input), "IQ4_NL/Q5_1 tiled LM Head tail reference");
}

static ActivationBuffer iq4_expert_reference(const TensorData& gate, const TensorData* up, const TensorData& down,
                                             const ActivationBuffer& input, bool interleaved, float limit)
{
    const auto projected = iq4_reference(gate, input);
    const auto projected_up = up ? iq4_reference(*up, input) : ActivationBuffer{};
    const uint32_t intermediate = down.shape[1];
    ActivationBuffer activated(input.rows(), intermediate);
    for (size_t token = 0; token < input.rows(); ++token)
        for (uint32_t column = 0; column < intermediate; ++column)
        {
            const uint32_t gate_column = up ? column : interleaved ? column * 2
                                                                   : column;
            const uint32_t up_column = interleaved ? column * 2 + 1 : intermediate + column;
            const float gate_value = projected.row(token)[gate_column];
            const float up_value = up ? projected_up.row(token)[column] : projected.row(token)[up_column];
            const float g = limit > 0 ? std::min(gate_value, limit) : gate_value;
            const float u = limit > 0 ? std::clamp(up_value, -limit, limit) : up_value;
            activated.row(token)[column] = g / (1.f + std::exp(-g)) * u;
        }
    return iq4_reference(down, activated);
}

static std::shared_ptr<DeviceTensor_vulkan> iq4_upload(const ActivationBuffer& input, uint32_t device, const VulkanRuntimePtr& runtime)
{
    TensorData seed;
    seed.dtype = DType::Float32;
    seed.shape = {input.columns(), input.columns()};
    seed.float32_data.assign(static_cast<size_t>(input.columns()) * input.columns(), 0.f);
    auto operation = Linear::create(seed, nullptr, device, runtime, OptimizationDefaultFlags);
    iq4_check(static_cast<bool>(operation), "IQ4_NL/Q5_1 resident input bridge");
    auto graph = CommandGraph_vulkan::create(*operation);
    auto input_device = std::make_shared<DeviceTensor_vulkan>();
    iq4_check(graph && graph->upload(input, *input_device) && graph->submit() && graph->wait(), "IQ4_NL/Q5_1 resident input upload");
    return input_device;
}

static void iq4_experts(uint32_t device, DType dtype, DType down_dtype)
{
    const uint32_t hidden = dtype == DType::Q4K ? 256 : 64;
    constexpr uint32_t intermediate = 32;
    const auto runtime = create_vulkan_runtime();
    const auto gate = std::make_shared<TensorData>(block_matrix(dtype, intermediate, hidden, 1));
    const auto up = std::make_shared<TensorData>(block_matrix(dtype, intermediate, hidden, 2));
    const auto down = std::make_shared<TensorData>(block_matrix(down_dtype, hidden, intermediate, 3));
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    iq4_check(static_cast<bool>(context), "IQ4_NL/Q5_1 Expert context");
    const uint64_t alignment = context->device()->info.buffer_offset_alignment();
    const uint64_t storage = QnkLinear_vulkan::storage_bytes(*gate, alignment)
                             + QnkLinear_vulkan::storage_bytes(*up, alignment)
                             + QnkLinear_vulkan::storage_bytes(*down, alignment);
    for (float limit : {0.0f, 0.25f})
    {
        const ExpertActivation activation = limit > 0 ? ExpertActivation::DeepSeekSwiGlu : ExpertActivation::Silu;
        auto separate = QnkExpert_vulkan::create_separate(*gate, nullptr, *up, nullptr, *down, nullptr, limit, device,
                                                          activation, runtime, OptimizationDefaultFlags);
        iq4_check(separate && separate->device_bytes() == storage, "IQ4_NL/Q5_1 separate Expert retains native bytes");
        for (size_t tokens : {size_t(1), size_t(4), size_t(5)})
        {
            const auto input = iq4_input(tokens, hidden);
            const auto expected = iq4_expert_reference(*gate, up.get(), *down, input, false, limit);
            ActivationBuffer actual;
            iq4_check(separate->forward(input, actual), "IQ4_NL/Q5_1 host-input Expert");
            iq4_compare(actual, expected, "IQ4_NL/Q5_1 Shared host reference");
            auto resident = iq4_upload(input, device, runtime);
            auto result = std::make_shared<DeviceTensor_vulkan>();
            const auto before = get_vulkan_statistics(runtime);
            iq4_check(separate->forward_device(*resident, *result), "IQ4_NL/Q5_1 Shared resident Expert");
            const auto after = get_vulkan_statistics(runtime);
            iq4_check(after.compute_submissions == before.compute_submissions + 1
                          && after.batch_uploads == before.batch_uploads && after.batch_downloads == before.batch_downloads,
                      "IQ4_NL/Q5_1 Shared chain has one submit without activation transfers");
            std::shared_ptr<DeviceTensor_vulkan> combined;
            iq4_check(MoeCombine_vulkan::forward({}, tokens, hidden, nullptr, result, combined), "IQ4_NL/Q5_1 Shared enters device Combine");
            iq4_check(separate->materialize(*combined, actual), "IQ4_NL/Q5_1 materialize combined output");
            iq4_compare(actual, expected, "IQ4_NL/Q5_1 resident Shared/Combine reference");
            iq4_check(separate->materialize(*resident, actual), "IQ4_NL/Q5_1 input remains owned");
            iq4_compare(actual, input, "IQ4_NL/Q5_1 resident input immutable");
            iq4_check(!separate->forward_device(*result, *result), "IQ4_NL/Q5_1 rejects aliased device output");
            DeviceTensor_vulkan empty;
            iq4_check(!separate->forward_device(empty, *result) && !result->empty(), "IQ4_NL/Q5_1 failed device call preserves prior output");
        }
    }
    auto packed = std::make_shared<TensorData>(block_matrix(dtype, intermediate * 2, hidden, 4));
    for (bool interleaved : {false, true})
    {
        packed->qnk_interleave_rows = interleaved;
        auto operation = QnkExpert_vulkan::create(*packed, nullptr, *down, nullptr, 0.0f, device,
                                                  ExpertActivation::Silu, runtime, OptimizationDefaultFlags);
        const auto input = iq4_input(5, hidden);
        ActivationBuffer actual;
        iq4_check(operation && operation->forward(input, actual), "IQ4_NL/Q5_1 packed gate/up Expert");
        iq4_compare(actual, iq4_expert_reference(*packed, nullptr, *down, input, interleaved, 0.0f), "IQ4_NL/Q5_1 packed Expert reference");
    }
    auto backend = create_vulkan_expert_backend(storage + 4096, device, nullptr, runtime, OptimizationDefaultFlags);
    std::shared_ptr<const void> pin;
    iq4_check(backend && backend->prepare_demand_separate("gguf-blocks", gate, nullptr, up, nullptr, down, nullptr, 0, 0.0f, ExpertActivation::Silu, pin),
              "IQ4_NL/Q5_1 native Expert cache admission");
    iq4_check(pin && backend->statistics().resident_size == storage, "IQ4_NL/Q5_1 cache accounts compact native bytes");
    const auto input = iq4_input(4, hidden);
    ActivationBuffer actual;
    const std::array<ExpertBackendRequest, 1> requests = {ExpertBackendRequest{"gguf-blocks", &input, &actual}};
    auto submission = backend->submit_batch(requests);
    iq4_check(static_cast<bool>(submission), "IQ4_NL/Q5_1 native Expert cache submission");
    const auto results = submission->wait();
    iq4_check(results.size() == 1 && results[0] == ExpertBackendExecutionResult::Executed && submission->commit(),
              "IQ4_NL/Q5_1 cached Expert executes on Vulkan");
    iq4_compare(actual, iq4_expert_reference(*gate, up.get(), *down, input, false, 0.0f), "IQ4_NL/Q5_1 cached Expert reference");

    const auto source = iq4_input(5, hidden);
    auto resident = iq4_upload(source, device, runtime);
    auto wave_backend = create_vulkan_expert_backend(storage * 2, device, nullptr, runtime, OptimizationDefaultFlags);
    iq4_check(static_cast<bool>(wave_backend), "native device Expert cache backend");
    const std::array<ExpertRoute, 3> first_routes = {{{4, 0, 0.7f}, {1, 1, 0.25f}, {4, 2, -0.1f}}};
    const std::array<ExpertRoute, 2> second_routes = {{{0, 0, 0.5f}, {1, 1, 0.4f}}};
    ActivationBuffer first_host;
    ActivationBuffer second_host;
    std::shared_ptr<DeviceTensor_vulkan> first_device;
    std::shared_ptr<DeviceTensor_vulkan> second_device;
    std::array<ExpertBackendRequest, 2> device_requests;
    device_requests[0].key = "native-device-first";
    device_requests[0].output = &first_host;
    device_requests[0].device_input = resident;
    device_requests[0].device_routes = first_routes;
    device_requests[0].device_output = &first_device;
    device_requests[0].input_rows = first_routes.size();
    device_requests[0].input_columns = hidden;
    device_requests[1] = device_requests[0];
    device_requests[1].key = "native-device-second";
    device_requests[1].output = &second_host;
    device_requests[1].device_routes = second_routes;
    device_requests[1].device_output = &second_device;
    device_requests[1].input_rows = second_routes.size();
    auto cold = wave_backend->submit_batch(device_requests);
    iq4_check(static_cast<bool>(cold), "device-only cold cache lookup");
    const auto cold_results = cold->wait();
    iq4_check(cold_results.size() == 2 && cold_results[0] == ExpertBackendExecutionResult::NotResident
                  && cold_results[1] == ExpertBackendExecutionResult::NotResident && !first_device && !second_device,
              "native cold miss publishes no host or device result");
    cold.reset();
    std::array<ExpertDemandRequest, 2> demands;
    for (size_t index = 0; index < demands.size(); ++index)
    {
        demands[index].key = device_requests[index].key;
        demands[index].gate_up = gate;
        demands[index].up = up;
        demands[index].down = down;
        demands[index].activation = ExpertActivation::Silu;
    }
    std::array<std::shared_ptr<const void>, 2> pins;
    auto upload = wave_backend->begin_demand_batch(demands);
    iq4_check(upload && upload->wait(pins) == 2 && pins[0] && pins[1], "cold native Expert demand upload retains both entries");
    upload.reset();
    const auto before_device = get_vulkan_statistics(runtime);
    auto device_submission = wave_backend->submit_batch(device_requests);
    iq4_check(static_cast<bool>(device_submission), "device-only routed native Expert submission");
    pins = {};
    const auto device_results = device_submission->wait();
    iq4_check(device_results.size() == 2 && device_results[0] == ExpertBackendExecutionResult::Executed
                  && device_results[1] == ExpertBackendExecutionResult::Executed && device_submission->commit(),
              "native Expert device batch commits after upload pins release");
    device_submission.reset();
    const auto after_device = get_vulkan_statistics(runtime);
    iq4_check(first_device && second_device && first_host.rows() == 0 && second_host.rows() == 0
                  && after_device.compute_submissions == before_device.compute_submissions + 1
                  && after_device.batch_uploads == before_device.batch_uploads
                  && after_device.batch_downloads == before_device.batch_downloads,
              "routed native GPU gather and Expert batch perform no activation transfer or CPU output");
    const auto reference_for = [&](std::span<const ExpertRoute> routes) {
        ActivationBuffer gathered(routes.size(), hidden);
        for (size_t row = 0; row < routes.size(); ++row)
            std::copy_n(source.row(routes[row].token_index), hidden, gathered.row(row));
        return iq4_expert_reference(*gate, up.get(), *down, gathered, false, 0.0f);
    };
    const auto first_expected = reference_for(first_routes);
    const auto second_expected = reference_for(second_routes);
    iq4_check(MoeCombine_vulkan::materialize(*first_device, actual), "native routed first output materialization");
    iq4_compare(actual, first_expected, "native GPU gather handles permuted and repeated source rows");
    const std::array<MoeCombineInput_vulkan, 2> combine_inputs = {{{nullptr, first_device, first_routes}, {nullptr, second_device, second_routes}}};
    std::shared_ptr<DeviceTensor_vulkan> combined;
    iq4_check(MoeCombine_vulkan::forward(combine_inputs, source.rows(), hidden, nullptr, {}, combined)
                  && MoeCombine_vulkan::materialize(*combined, actual),
              "native routed results enter weighted GPU Combine");
    ActivationBuffer combined_expected(source.rows(), hidden);
    for (size_t row = 0; row < first_routes.size(); ++row)
        for (uint32_t column = 0; column < hidden; ++column)
            combined_expected.row(first_routes[row].token_index)[column] += first_routes[row].weight * first_expected.row(row)[column];
    for (size_t row = 0; row < second_routes.size(); ++row)
        for (uint32_t column = 0; column < hidden; ++column)
            combined_expected.row(second_routes[row].token_index)[column] += second_routes[row].weight * second_expected.row(row)[column];
    iq4_compare(actual, combined_expected, "native GPU weighted route reduction reference");
    std::shared_ptr<const void> replacement_pin;
    iq4_check(wave_backend->prepare_demand_separate("native-device-replacement", gate, nullptr, up, nullptr, down, nullptr,
                                                    0, 0.0f, ExpertActivation::Silu, replacement_pin)
                  && wave_backend->statistics().evictions != 0,
              "native cold demand evicts a completed entry");
    iq4_check(MoeCombine_vulkan::materialize(*first_device, actual), "completed native activation survives weight cache eviction");
    iq4_compare(actual, first_expected, "native output lifetime is independent of cache Entry pins");
    const std::array<ExpertRoute, 1> invalid_routes = {{{5, 0, 1.f}}};
    ExpertBackendRequest invalid_request = device_requests[0];
    invalid_request.key = "native-device-replacement";
    invalid_request.device_routes = invalid_routes;
    invalid_request.input_rows = invalid_routes.size();
    const auto preserved_output = first_device;
    auto invalid_submission = wave_backend->submit_batch({&invalid_request, 1});
    iq4_check(static_cast<bool>(invalid_submission), "malformed device route submission is contained");
    const auto invalid_results = invalid_submission->wait();
    invalid_submission->abort();
    iq4_check(invalid_results.size() == 1 && invalid_results[0] == ExpertBackendExecutionResult::Failed
                  && first_device == preserved_output,
              "invalid native gather fails without publishing or modifying prior output");
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
        for (uint32_t device = 0; device < ncnn::moe::get_gpu_count(); ++device)
        {
            for (ncnn::moe::DType dtype : {ncnn::moe::DType::IQ4NL, ncnn::moe::DType::Q5_1})
            {
                ncnn::moe::iq4_projections(device, dtype);
                ncnn::moe::iq4_experts(device, dtype, dtype);
            }
            // The real Qwen package mixes Q4_K gate/up and Q5_1 down;
            // Q8_0 plus Q5_1 remains useful independent mixed-block coverage.
            ncnn::moe::iq4_experts(device, ncnn::moe::DType::Q4K, ncnn::moe::DType::Q5_1);
            ncnn::moe::iq4_experts(device, ncnn::moe::DType::Q8_0, ncnn::moe::DType::Q5_1);
        }
        std::cout << "Native IQ4_NL/Q5_1 Vulkan projection, resident Shared chain and Expert cache tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Native IQ4_NL/Q5_1 Vulkan test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
