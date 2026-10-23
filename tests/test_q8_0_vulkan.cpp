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
    q8_check(backend && backend->prepare_demand_separate("q8-0", gate, nullptr, up, nullptr, down, nullptr, 0, limit, ExpertActivation::DeepSeekSwiGlu, pin),
             "Q8_0 native Expert cache admission");
    q8_check(pin && backend->statistics().resident_size == storage, "Q8_0 Expert cache accounts compact native storage");
    const auto input = q8_input(4, hidden);
    ActivationBuffer actual;
    const std::array<ExpertBackendRequest, 1> requests = {ExpertBackendRequest{"q8-0", &input, &actual}};
    auto submission = backend->submit_batch(requests);
    q8_check(static_cast<bool>(submission), "Q8_0 native Expert cache submission");
    const auto results = submission->wait();
    q8_check(results.size() == 1 && results[0] == ExpertBackendExecutionResult::Executed && submission->commit(),
             "Q8_0 cached Expert executes on Vulkan");
    q8_compare(actual, q8_expert_reference(*gate, up.get(), *down, input, false, limit), "Q8_0 cached Expert reference");
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
