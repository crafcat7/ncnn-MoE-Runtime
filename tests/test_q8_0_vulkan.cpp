#include "backends/ncnn/expertbackend_vulkan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/moecombine_vulkan.h"
#include "backends/ncnn/vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "engine/expertbackend.h"
#include "kernels/bfloat16.h"
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
static void q8_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static TensorData q8_matrix(uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData matrix;
    matrix.dtype = DType::Q8_0;
    matrix.shape = {rows, columns};
    const size_t size = static_cast<size_t>(qnk_storage_bytes(matrix.dtype, rows, columns));
    q8_check(size == static_cast<size_t>(rows) * (columns / 32) * 34, "Q8_0 retains 34-byte blocks");
    // Deliberately unaligned host mapping and odd block counts also exercise
    // the shader's unaligned half-scale and signed-byte reads across rows.
    std::shared_ptr<uint8_t[]> owner(new uint8_t[size + 1]);
    uint8_t* bytes = owner.get() + 1;
    for (size_t block = 0; block < size / 34; ++block)
    {
        const bool negative = (block + seed) % 3 == 1;
        bytes[block * 34] = 0;
        bytes[block * 34 + 1] = negative ? 0x98 : 0x18;
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const int code = lane == 0 ? -128 : lane == 1 ? 127
                                                          : static_cast<int>((block * 17 + lane * 29 + seed * 11) % 255) - 127;
            bytes[block * 34 + 2 + lane] = static_cast<uint8_t>(code);
        }
    }
    matrix.mapped_data = std::shared_ptr<const uint8_t>(owner, bytes);
    matrix.mapped_size = size;
    return matrix;
}

static ActivationBuffer q8_input(size_t rows, uint32_t columns)
{
    ActivationBuffer input(rows, columns);
    for (size_t token = 0; token < rows; ++token)
        for (uint32_t column = 0; column < columns; ++column)
            input.row(token)[column] = static_cast<float>(static_cast<int>((token * 13 + column * 7 + column / 5) % 41) - 20) / 32.f;
    return input;
}

static ActivationBuffer q8_reference(const TensorData& matrix, const ActivationBuffer& input, uint32_t groups = 1)
{
    ActivationBuffer output(input.rows(), matrix.shape[0]);
    const uint32_t columns = matrix.shape[1];
    const uint32_t blocks = columns / 32;
    const auto raw = matrix.qnk_values();
    for (size_t token = 0; token < input.rows(); ++token)
        for (uint32_t row = 0; row < matrix.shape[0]; ++row)
        {
            const uint32_t group = row / (matrix.shape[0] / groups);
            double sum = 0;
            for (uint32_t column = 0; column < columns; ++column)
            {
                const size_t offset = (static_cast<size_t>(row) * blocks + column / 32) * 34;
                const int byte = raw[offset + 2 + column % 32];
                const int code = byte < 128 ? byte : byte - 256;
                const float scale = raw[offset + 1] == 0x98 ? -1.f / 512.f : 1.f / 512.f;
                sum += static_cast<double>(code) * scale * input.row(token)[group * columns + column];
            }
            output.row(token)[row] = static_cast<float>(sum);
        }
    return output;
}

static void q8_compare(const ActivationBuffer& actual, const ActivationBuffer& expected, const std::string& label)
{
    q8_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + ": shape");
    for (size_t index = 0; index < expected.values().size(); ++index)
    {
        const float reference = expected.values()[index];
        q8_check(std::isfinite(actual.values()[index])
                     && std::abs(actual.values()[index] - reference) <= 0.00002f + 0.00003f * std::abs(reference),
                 label + ": element " + std::to_string(index) + " actual=" + std::to_string(actual.values()[index])
                     + " expected=" + std::to_string(reference));
    }
}

static void q8_projections(uint32_t device)
{
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    q8_check(static_cast<bool>(context), "Q8_0 projection context");
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    const auto align = [alignment](uint64_t value) { return (value + alignment - 1) / alignment * alignment; };
    for (uint32_t columns : {32u, 64u, 288u})
        for (uint32_t groups : {1u, 2u})
        {
            const TensorData matrix = q8_matrix(groups == 1 ? 5 : 6, columns, columns);
            auto operation = QnkLinear_vulkan::create(matrix, nullptr, device, runtime, OptimizationDefaultFlags, groups);
            const uint64_t expected_storage = align(matrix.qnk_values().size()) + align(matrix.shape[0] * sizeof(float));
            q8_check(operation && operation->dtype() == DType::Q8_0
                         && operation->device_bytes() == expected_storage
                         && QnkLinear_vulkan::storage_bytes(matrix, alignment) == expected_storage,
                     "Q8_0 GPU storage contains raw blocks and bias only");
            for (size_t tokens : {size_t(1), size_t(4), size_t(5)})
            {
                const auto input = q8_input(tokens, columns * groups);
                const auto expected = q8_reference(matrix, input, groups);
                ActivationBuffer actual;
                const auto before = get_vulkan_statistics(runtime);
                q8_check(operation->forward(input, actual), "Q8_0 native projection dispatch");
                q8_check(get_vulkan_statistics(runtime).dispatches == before.dispatches + 1,
                         "Q8_0 performs one native projection");
                q8_compare(actual, expected, "Q8_0 grouped/block-tail GPU reference");
                if (groups == 1)
                    q8_compare(forward_linear(matrix, input, OptimizationDefaultFlags), expected, "Q8_0 CPU native reference");
            }
        }
    TensorData invalid = q8_matrix(5, 32, 1);
    invalid.shape[1] = 33;
    q8_check(!QnkLinear_vulkan::create(invalid, nullptr, device, runtime, OptimizationDefaultFlags), "reject incomplete Q8_0 block");
    const TensorData vocabulary = q8_matrix(65537, 32, 1);
    auto head = QnkLinear_vulkan::create(vocabulary, nullptr, device, runtime, OptimizationDefaultFlags);
    const auto input = q8_input(1, 32);
    ActivationBuffer logits;
    q8_check(head && head->forward(input, logits), "Q8_0 vocabulary projection exceeds 65535 workgroups");
    q8_compare(logits, q8_reference(vocabulary, input), "Q8_0 tiled LM Head tail reference");
}

static ActivationBuffer q8_expert_reference(const TensorData& gate, const TensorData* up, const TensorData& down,
                                            const ActivationBuffer& input, bool interleaved, float limit)
{
    const auto projected = q8_reference(gate, input);
    const auto projected_up = up ? q8_reference(*up, input) : ActivationBuffer{};
    const uint32_t intermediate = down.shape[1];
    ActivationBuffer activated(input.rows(), intermediate);
    for (size_t token = 0; token < input.rows(); ++token)
        for (uint32_t column = 0; column < intermediate; ++column)
        {
            const uint32_t gate_column = up ? column : interleaved ? column * 2
                                                                   : column;
            const uint32_t up_column = interleaved ? column * 2 + 1 : intermediate + column;
            const float g = std::min(projected.row(token)[gate_column], limit);
            const float u = std::clamp(up ? projected_up.row(token)[column] : projected.row(token)[up_column], -limit, limit);
            activated.row(token)[column] = g / (1.f + std::exp(-g)) * u;
        }
    return q8_reference(down, activated);
}

static std::shared_ptr<DeviceTensor_vulkan> q8_upload(const ActivationBuffer& input, uint32_t device, const VulkanRuntimePtr& runtime)
{
    TensorData seed;
    seed.dtype = DType::Float32;
    seed.shape = {input.columns(), input.columns()};
    seed.float32_data.assign(static_cast<size_t>(input.columns()) * input.columns(), 0.f);
    auto operation = Linear::create(seed, nullptr, device, runtime, OptimizationDefaultFlags);
    q8_check(static_cast<bool>(operation), "Q8_0 resident input bridge");
    auto graph = CommandGraph_vulkan::create(*operation);
    auto input_device = std::make_shared<DeviceTensor_vulkan>();
    q8_check(graph && graph->upload(input, *input_device) && graph->submit() && graph->wait(), "Q8_0 resident input upload");
    return input_device;
}

static void q8_experts(uint32_t device)
{
    constexpr uint32_t hidden = 64;
    constexpr uint32_t intermediate = 32;
    constexpr float limit = 0.25f;
    const auto runtime = create_vulkan_runtime();
    const auto gate = std::make_shared<TensorData>(q8_matrix(intermediate, hidden, 1));
    const auto up = std::make_shared<TensorData>(q8_matrix(intermediate, hidden, 2));
    const auto down = std::make_shared<TensorData>(q8_matrix(hidden, intermediate, 3));
    auto separate = QnkExpert_vulkan::create_separate(*gate, nullptr, *up, nullptr, *down, nullptr, limit, device,
                                                      ExpertActivation::DeepSeekSwiGlu, runtime, OptimizationDefaultFlags);
    q8_check(static_cast<bool>(separate), "Q8_0 separate Shared Expert construction");
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    const uint64_t alignment = context->device()->info.buffer_offset_alignment();
    const uint64_t storage = QnkLinear_vulkan::storage_bytes(*gate, alignment)
                             + QnkLinear_vulkan::storage_bytes(*up, alignment)
                             + QnkLinear_vulkan::storage_bytes(*down, alignment);
    q8_check(separate->device_bytes() == storage, "Q8_0 separate gate/up/down retains native bytes");
    for (size_t tokens : {size_t(1), size_t(4), size_t(5)})
    {
        const auto input = q8_input(tokens, hidden);
        const auto expected = q8_expert_reference(*gate, up.get(), *down, input, false, limit);
        ActivationBuffer actual;
        q8_check(separate->forward(input, actual), "Q8_0 separate host Expert");
        q8_compare(actual, expected, "Q8_0 Shared host reference");
        auto resident = q8_upload(input, device, runtime);
        auto result = std::make_shared<DeviceTensor_vulkan>();
        const auto before = get_vulkan_statistics(runtime);
        q8_check(separate->forward_device(*resident, *result), "Q8_0 Shared resident Expert");
        const auto after = get_vulkan_statistics(runtime);
        q8_check(after.compute_submissions == before.compute_submissions + 1
                     && after.batch_uploads == before.batch_uploads && after.batch_downloads == before.batch_downloads,
                 "Q8_0 Shared chain has one submit without activation transfers");
        std::shared_ptr<DeviceTensor_vulkan> combined;
        q8_check(MoeCombine_vulkan::forward({}, tokens, hidden, nullptr, result, combined), "Q8_0 Shared enters device Combine");
        q8_check(separate->materialize(*combined, actual), "Q8_0 materialize combined output");
        q8_compare(actual, expected, "Q8_0 resident Shared/Combine reference");
        q8_check(separate->materialize(*resident, actual), "Q8_0 input remains owned");
        q8_compare(actual, input, "Q8_0 resident input immutable");
        q8_check(!separate->forward_device(*result, *result), "Q8_0 rejects aliased device output");
        DeviceTensor_vulkan empty;
        q8_check(!separate->forward_device(empty, *result) && !result->empty(), "Q8_0 failed device call preserves prior output");
    }
    auto packed = std::make_shared<TensorData>(q8_matrix(intermediate * 2, hidden, 4));
    for (bool interleaved : {false, true})
    {
        packed->qnk_interleave_rows = interleaved;
        auto operation = QnkExpert_vulkan::create(*packed, nullptr, *down, nullptr, limit, device,
                                                  ExpertActivation::DeepSeekSwiGlu, runtime, OptimizationDefaultFlags);
        const auto input = q8_input(5, hidden);
        ActivationBuffer actual;
        q8_check(operation && operation->forward(input, actual), "Q8_0 packed gate/up Expert");
        q8_compare(actual, q8_expert_reference(*packed, nullptr, *down, input, interleaved, limit), "Q8_0 packed Expert reference");
    }
    auto backend = create_vulkan_expert_backend(storage + 4096, device, nullptr, runtime, OptimizationDefaultFlags);
    std::shared_ptr<const void> pin;
    q8_check(backend && backend->prepare_demand(ExpertDemandRequest{"q8-0", gate, nullptr, down, nullptr, 0, limit, ExpertActivation::DeepSeekSwiGlu, up}, pin),
             "Q8_0 native Expert cache admission");
    q8_check(pin && backend->statistics().resident_size == storage, "Q8_0 Expert cache accounts compact native storage");
    const auto input = q8_input(4, hidden);
    ActivationBuffer actual;
    const std::array<ExpertRequest, 1> requests = {ExpertRequest{"q8-0", &input, &actual}};
    auto submission = backend->submit_batch(requests);
    q8_check(static_cast<bool>(submission), "Q8_0 native Expert cache submission");
    const auto results = submission->wait();
    q8_check(results.size() == 1 && results[0] == ExpertStatus::Executed && submission->commit(),
             "Q8_0 cached Expert executes on Vulkan");
    q8_compare(actual, q8_expert_reference(*gate, up.get(), *down, input, false, limit), "Q8_0 cached Expert reference");

    TensorData router;
    router.dtype = DType::Float32;
    router.shape = {1, hidden};
    router.float32_data.resize(hidden);
    for (uint32_t column = 0; column < hidden; ++column)
        router.float32_data[column] = static_cast<float>(static_cast<int>(column % 9) - 4) / 8.f;
    q8_check(separate->prepare_router_gate(router) && separate->has_router_gate(), "Q8_0 Shared router gate prepared on device");
    TensorData invalid_router = router;
    invalid_router.shape = {2, hidden};
    q8_check(!separate->prepare_router_gate(invalid_router) && separate->has_router_gate(), "invalid router preparation preserves prior gate");
    for (size_t tokens : {size_t(1), size_t(5)})
    {
        const auto gated_input = q8_input(tokens, hidden);
        auto expected = q8_expert_reference(*gate, up.get(), *down, gated_input, false, limit);
        for (size_t token = 0; token < tokens; ++token)
        {
            double logit = 0;
            for (uint32_t column = 0; column < hidden; ++column)
                logit += static_cast<double>(router.float32_data[column]) * gated_input.row(token)[column];
            const float scale = 1.f / (1.f + std::exp(-static_cast<float>(logit)));
            for (uint32_t column = 0; column < hidden; ++column)
                expected.row(token)[column] *= scale;
        }
        auto resident = q8_upload(gated_input, device, runtime);
        DeviceTensor_vulkan result;
        const auto before = get_vulkan_statistics(runtime);
        q8_check(separate->forward_device(*resident, result), "Q8_0 router-gated Shared device chain");
        const auto after = get_vulkan_statistics(runtime);
        q8_check(after.compute_submissions == before.compute_submissions + 1
                     && after.batch_uploads == before.batch_uploads && after.batch_downloads == before.batch_downloads,
                 "Q8_0 router gate stays in one device submission without activation transfers");
        q8_check(separate->materialize(result, actual), "Q8_0 gated output materialization");
        q8_compare(actual, expected, "Q8_0 GPU router sigmoid and scale reference");
    }
}

static void float32_device_expert(uint32_t device, bool separate, bool interleaved)
{
    constexpr uint32_t hidden = 64;
    constexpr uint32_t intermediate = 32;
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    q8_check(static_cast<bool>(context), "FP32 routed Expert context");
    const auto matrix = [](uint32_t rows, uint32_t columns, uint32_t seed) {
        auto weight = std::make_shared<TensorData>();
        weight->dtype = DType::Float32;
        weight->shape = {rows, columns};
        weight->float32_data.resize(static_cast<size_t>(rows) * columns);
        for (size_t index = 0; index < weight->float32_data.size(); ++index)
            weight->float32_data[index] = static_cast<float>(static_cast<int>((index * 7 + seed * 3) % 19) - 9) / 128.f;
        return weight;
    };
    auto gate = matrix(separate ? intermediate : intermediate * 2, hidden, 1);
    gate->qnk_interleave_rows = interleaved;
    auto up = separate ? matrix(intermediate, hidden, 3) : nullptr;
    auto down = matrix(hidden, intermediate, 2);
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    const uint64_t expected_charge = QnkLinear_vulkan::storage_bytes(*gate, alignment)
                                     + QnkLinear_vulkan::storage_bytes(*down, alignment)
                                     + (up ? QnkLinear_vulkan::storage_bytes(*up, alignment) : 0);
    q8_check(expected_charge != 0
                 && native_quantized_expert_storage_bytes(*gate, up.get(), *down, device, runtime, OptimizationDefaultFlags) == expected_charge,
             "FP32 packed and separate storage planning retain original FP32 projections");
    TensorData truncated = *gate;
    truncated.float32_data.pop_back();
    q8_check(native_quantized_expert_storage_bytes(truncated, up.get(), *down, device, runtime, OptimizationDefaultFlags) == 0,
             "FP32 storage planning rejects a truncated original matrix");
    auto backend = create_vulkan_expert_backend(expected_charge, device, nullptr, runtime, OptimizationDefaultFlags);
    const auto source = q8_input(5, hidden);
    auto resident = q8_upload(source, device, runtime);
    const std::array<ExpertRoute, 3> routes = {{{4, 0, 0.7f}, {1, 1, 0.2f}, {4, 2, -0.1f}}};
    ExpertDemandRequest demand;
    demand.key = separate ? "fp32-separate-device" : "fp32-packed-device";
    demand.gate_up = gate;
    demand.up = up;
    demand.down = down;
    demand.activation = ExpertActivation::Silu;
    auto upload = backend->begin_demand_batch({&demand, 1});
    std::array<std::shared_ptr<const void>, 1> pins;
    q8_check(upload && upload->wait(pins) == 1 && pins.front(), "FP32 cold device Expert admission");
    upload.reset();
    q8_check(backend->statistics().resident_size == expected_charge
                 && gate->bfloat16_values().empty() && down->bfloat16_values().empty()
                 && (!up || up->bfloat16_values().empty()),
             "FP32 cache admission matches exact native storage without BF16 conversion");
    ActivationBuffer actual;
    std::shared_ptr<DeviceTensor_vulkan> output;
    ExpertRequest request;
    request.key = demand.key;
    request.output = &actual;
    request.device_input = resident;
    request.device_routes = routes;
    request.device_output = &output;
    request.input_rows = routes.size();
    request.input_columns = hidden;
    const auto before = get_vulkan_statistics(runtime);
    auto submission = backend->submit_batch({&request, 1});
    pins = {};
    q8_check(static_cast<bool>(submission), "FP32 device-only Expert request");
    const auto results = submission->wait();
    q8_check(results.size() == 1 && results.front() == ExpertStatus::Executed && submission->commit(),
             "FP32 routed device Expert commits without host input");
    const auto after = get_vulkan_statistics(runtime);
    q8_check(output && actual.rows() == 0 && after.compute_submissions == before.compute_submissions + 1
                 && after.batch_uploads == before.batch_uploads && after.batch_downloads == before.batch_downloads,
             "FP32 gather, projections and activation remain on device");
    ActivationBuffer gathered(routes.size(), hidden);
    for (size_t row = 0; row < routes.size(); ++row)
        std::copy_n(source.row(routes[row].token_index), hidden, gathered.row(row));
    const auto projected = forward_linear(*gate, gathered, OptimizationDefaultFlags);
    const auto projected_up = up ? forward_linear(*up, gathered, OptimizationDefaultFlags) : ActivationBuffer();
    ActivationBuffer activated(routes.size(), intermediate);
    for (size_t row = 0; row < routes.size(); ++row)
        for (uint32_t column = 0; column < intermediate; ++column)
        {
            const uint32_t gate_column = up || !interleaved ? column : column * 2;
            const uint32_t up_column = interleaved ? column * 2 + 1 : intermediate + column;
            const float value = projected.row(row)[gate_column];
            activated.row(row)[column] = value / (1.f + std::exp(-value))
                                         * (up ? projected_up.row(row)[column] : projected.row(row)[up_column]);
        }
    q8_check(MoeCombine_vulkan::materialize(*output, actual), "FP32 routed device output materialization");
    q8_compare(actual, forward_linear(*down, activated, OptimizationDefaultFlags),
               separate ? "FP32 separate device Expert CPU reference" : interleaved ? "FP32 interleaved device Expert CPU reference"
                                                                                    : "FP32 packed device Expert CPU reference");
}

static void bfloat16_device_expert(uint32_t device)
{
    constexpr uint32_t hidden = 128;
    constexpr uint32_t intermediate = 128;
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    q8_check(static_cast<bool>(context), "BF16 routed Expert context");
    if (context->device()->info.subgroup_size() != 32)
    {
        std::cout << "SKIP optional BF16 routed Expert device " << device << ": backend requires subgroup size 32\n";
        return;
    }
    const auto matrix = [](uint32_t rows, uint32_t columns, uint32_t seed) {
        auto weight = std::make_shared<TensorData>();
        weight->dtype = DType::BFloat16;
        weight->shape = {rows, columns};
        weight->bfloat16_data.resize(static_cast<size_t>(rows) * columns);
        for (size_t index = 0; index < weight->bfloat16_data.size(); ++index)
            weight->bfloat16_data[index] = float_to_bfloat16(static_cast<float>(static_cast<int>((index * 7 + seed * 3) % 19) - 9) / 128.f);
        return weight;
    };
    auto gate_up = matrix(intermediate * 2, hidden, 1);
    auto down = matrix(hidden, intermediate, 2);
    auto backend = create_vulkan_expert_backend(1024 * 1024, device, nullptr, runtime, OptimizationDefaultFlags);
    const auto source = q8_input(5, hidden);
    auto resident = q8_upload(source, device, runtime);
    const std::array<ExpertRoute, 3> routes = {{{4, 0, 0.7f}, {1, 1, 0.2f}, {4, 2, -0.1f}}};
    ExpertDemandRequest demand;
    demand.key = "bf16-device";
    demand.gate_up = gate_up;
    demand.down = down;
    demand.activation = ExpertActivation::Silu;
    auto upload = backend->begin_demand_batch({&demand, 1});
    std::array<std::shared_ptr<const void>, 1> pins;
    q8_check(upload && upload->wait(pins) == 1 && pins.front(), "BF16 cold device Expert admission on device " + std::to_string(device));
    upload.reset();
    const uint64_t expected_charge = static_cast<uint64_t>(gate_up->bfloat16_data.size() + down->bfloat16_data.size()) * sizeof(uint16_t);
    q8_check(native_quantized_expert_storage_bytes(*gate_up, nullptr, *down, device, runtime, OptimizationDefaultFlags) == expected_charge
                 && backend->statistics().resident_size == expected_charge,
             "BF16 eager cache planning and admission share the same payload charge");
    ActivationBuffer actual;
    std::shared_ptr<DeviceTensor_vulkan> output;
    ExpertRequest request;
    request.key = demand.key;
    request.output = &actual;
    request.device_input = resident;
    request.device_routes = routes;
    request.device_output = &output;
    request.input_rows = routes.size();
    request.input_columns = hidden;
    const auto before = get_vulkan_statistics(runtime);
    auto submission = backend->submit_batch({&request, 1});
    pins = {};
    q8_check(static_cast<bool>(submission), "BF16 device-only Expert request");
    const auto results = submission->wait();
    q8_check(results.size() == 1 && results.front() == ExpertStatus::Executed && submission->commit(),
             "BF16 routed device Expert commits without host input");
    const auto after = get_vulkan_statistics(runtime);
    q8_check(output && actual.rows() == 0 && after.compute_submissions == before.compute_submissions + 1
                 && after.batch_uploads == before.batch_uploads && after.batch_downloads == before.batch_downloads,
             "BF16 GPU gather/projection/SwiGLUDown retains output without activation transfers");
    ActivationBuffer gathered(routes.size(), hidden);
    for (size_t row = 0; row < routes.size(); ++row)
        std::copy_n(source.row(routes[row].token_index), hidden, gathered.row(row));
    const auto projected = forward_linear(*gate_up, gathered, OptimizationDefaultFlags);
    ActivationBuffer activated(routes.size(), intermediate);
    for (size_t row = 0; row < routes.size(); ++row)
        for (uint32_t column = 0; column < intermediate; ++column)
        {
            const float gate = projected.row(row)[column];
            activated.row(row)[column] = gate / (1.f + std::exp(-gate)) * projected.row(row)[intermediate + column];
        }
    q8_check(MoeCombine_vulkan::materialize(*output, actual), "BF16 routed device output materialization");
    q8_compare(actual, forward_linear(*down, activated, OptimizationDefaultFlags), "BF16 device Expert CPU reference");
}

static void mxfp4_storage_billing(uint32_t device)
{
    const auto runtime = create_vulkan_runtime();
    const auto matrix = [](uint32_t rows, uint32_t columns) {
        auto weight = std::make_shared<TensorData>();
        weight->dtype = DType::MxFp4;
        weight->shape = {rows, columns};
        weight->mxfp4_blocks.resize(static_cast<size_t>(rows) * (columns / 32) * 16);
        weight->mxfp4_scales.resize(static_cast<size_t>(rows) * (columns / 32));
        std::fill_n(weight->mxfp4_blocks.data(), weight->mxfp4_blocks.size(), uint8_t(0x11));
        std::fill_n(weight->mxfp4_scales.data(), weight->mxfp4_scales.size(), uint8_t(127));
        return weight;
    };
    const auto gate = matrix(256, 128);
    const auto down = matrix(128, 128);
    TensorData gate_bias;
    gate_bias.dtype = DType::Float32;
    gate_bias.shape = {256};
    gate_bias.float32_data.assign(256, 0.25f);
    TensorData down_bias;
    down_bias.dtype = DType::Float32;
    down_bias.shape = {128};
    down_bias.float32_data.assign(128, -0.125f);
    const uint64_t payload = gate->mxfp4_blocks.size() + gate->mxfp4_scales.size()
                             + down->mxfp4_blocks.size() + down->mxfp4_scales.size();
    const uint64_t expected_charge = payload + (gate_bias.float32_data.size() + down_bias.float32_data.size()) * sizeof(float);
    q8_check(native_quantized_expert_storage_bytes(*gate, nullptr, *down, device, runtime, OptimizationDefaultFlags) == payload,
             "MXFP4 eager storage planning uses the original packed payload");
    q8_check(native_quantized_expert_storage_bytes(*gate, nullptr, *down, device, runtime, OptimizationDefaultFlags,
                                                   &gate_bias, nullptr, &down_bias, ExpertActivation::Silu, 0.f)
                 == expected_charge,
             "MXFP4 storage planning includes explicit biases");
    auto backend = create_vulkan_expert_backend(1024 * 1024, device, nullptr, runtime, OptimizationDefaultFlags);
    std::shared_ptr<const void> pin;
    q8_check(backend && backend->prepare_demand(ExpertDemandRequest{"mxfp4-billing", gate, &gate_bias, down, &down_bias, 0, 0.f, ExpertActivation::Silu}, pin)
                 && pin && backend->statistics().resident_size == expected_charge,
             "MXFP4 eager cache admission and size helper use identical accounting");
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
            ncnn::moe::q8_projections(device);
            ncnn::moe::q8_experts(device);
            ncnn::moe::bfloat16_device_expert(device);
            ncnn::moe::float32_device_expert(device, false, false);
            ncnn::moe::float32_device_expert(device, false, true);
            ncnn::moe::float32_device_expert(device, true, false);
            ncnn::moe::mxfp4_storage_billing(device);
        }
        std::cout << "Native Q8_0 Vulkan projection, resident Shared chain and Expert cache tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Native Q8_0 Vulkan test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
