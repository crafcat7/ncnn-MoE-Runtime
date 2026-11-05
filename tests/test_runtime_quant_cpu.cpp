#include "test_runtime_decls.h"
#include "test_runtime_support.h"

namespace ncnn {
namespace moe {

void test_gguf_mxfp4_row_unpack()
{
    // Decode individual nibbles to keep the oracle independent of SIMD packing.
    const auto reference = [](std::span<const uint8_t> source, size_t blocks_per_row,
                              size_t block_stride, size_t scale_stride,
                              std::span<uint8_t> blocks, std::span<uint8_t> scales) {
        const size_t rows = source.size() / 17 / blocks_per_row;
        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t block = 0; block < blocks_per_row; ++block)
            {
                const size_t source_offset = (row * blocks_per_row + block) * 17;
                scales[row * scale_stride + block] = source[source_offset];
                for (size_t value = 0; value < 32; ++value)
                {
                    const uint8_t code = static_cast<uint8_t>((source[source_offset + 1 + value % 16] >> (value / 16 * 4)) & 15u);
                    uint8_t& output = blocks[row * block_stride + block * 16 + value / 2];
                    const uint8_t mask = static_cast<uint8_t>(15u << (value % 2 * 4));
                    output = static_cast<uint8_t>((output & ~mask) | (code << (value % 2 * 4)));
                }
            }
        }
    };
    for (size_t blocks_per_row : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, size_t{31}, size_t{32}, size_t{33}, size_t{127}, size_t{128}, size_t{129}})
    {
        constexpr size_t rows = 9;
        const size_t source_size = rows * blocks_per_row * 17;
        std::vector<uint8_t> source_storage(source_size + 3, 0x5a);
        std::span<uint8_t> source(source_storage.data() + 1, source_size);
        for (size_t index = 0; index < source.size(); ++index)
            source[index] = static_cast<uint8_t>(index * 73 + index / 17 * 19);
        // All exponent bytes, including 255, are raw values rather than floats.
        for (size_t block = 0; block < rows * blocks_per_row; ++block)
            source[block * 17] = static_cast<uint8_t>(255 - block);
        const size_t block_stride = blocks_per_row * 32 + 7;
        const size_t scale_stride = blocks_per_row * 2 + 3;
        const size_t block_size = (rows - 1) * block_stride + blocks_per_row * 16;
        const size_t scale_size = (rows - 1) * scale_stride + blocks_per_row;
        std::vector<uint8_t> actual_blocks(block_size + 11, 0xa5);
        std::vector<uint8_t> actual_scales(scale_size + 13, 0x3c);
        std::vector<uint8_t> expected_blocks = actual_blocks;
        std::vector<uint8_t> expected_scales = actual_scales;
        reference(source, blocks_per_row, block_stride, scale_stride,
                  std::span<uint8_t>(expected_blocks).subspan(3, block_size), std::span<uint8_t>(expected_scales).subspan(5, scale_size));
        check(unpack_gguf_mxfp4_rows(source, blocks_per_row, block_stride, scale_stride,
                                     std::span<uint8_t>(actual_blocks).subspan(3, block_size), std::span<uint8_t>(actual_scales).subspan(5, scale_size)));
        check(actual_blocks == expected_blocks && actual_scales == expected_scales);

        // Sequential single-source calls populate even and odd rows directly.
        std::vector<uint8_t> interleaved_blocks(rows * blocks_per_row * 32, 0xa5);
        std::vector<uint8_t> interleaved_scales(rows * blocks_per_row * 2, 0x3c);
        std::vector<uint8_t> expected_interleaved_blocks = interleaved_blocks;
        std::vector<uint8_t> expected_interleaved_scales = interleaved_scales;
        for (size_t parity = 0; parity < 2; ++parity)
        {
            for (uint8_t& value : source) value ^= 0xf3;
            reference(source, blocks_per_row, blocks_per_row * 32, blocks_per_row * 2,
                      std::span<uint8_t>(expected_interleaved_blocks).subspan(parity * blocks_per_row * 16),
                      std::span<uint8_t>(expected_interleaved_scales).subspan(parity * blocks_per_row));
            check(unpack_gguf_mxfp4_rows(source, blocks_per_row, blocks_per_row * 32, blocks_per_row * 2,
                                         std::span<uint8_t>(interleaved_blocks).subspan(parity * blocks_per_row * 16),
                                         std::span<uint8_t>(interleaved_scales).subspan(parity * blocks_per_row)));
        }
        check(interleaved_blocks == expected_interleaved_blocks && interleaved_scales == expected_interleaved_scales);

        std::vector<uint8_t> contiguous_blocks(rows * blocks_per_row * 16, 0xa5);
        std::vector<uint8_t> contiguous_scales(rows * blocks_per_row, 0x3c);
        std::vector<uint8_t> expected_contiguous_blocks = contiguous_blocks;
        std::vector<uint8_t> expected_contiguous_scales = contiguous_scales;
        reference(source, rows * blocks_per_row, contiguous_blocks.size(), contiguous_scales.size(), expected_contiguous_blocks, expected_contiguous_scales);
        check(unpack_gguf_mxfp4(source, contiguous_blocks, contiguous_scales));
        check(contiguous_blocks == expected_contiguous_blocks && contiguous_scales == expected_contiguous_scales);
    }

    std::array<uint8_t, 256> storage;
    std::array<uint8_t, 64> blocks;
    std::array<uint8_t, 8> scales;
    storage.fill(0x39);
    blocks.fill(0xa5);
    scales.fill(0x3c);
    const auto source = std::span<const uint8_t>(storage).subspan(1, 68);
    const auto reject = [&](std::span<const uint8_t> input, size_t blocks_per_row, size_t block_stride, size_t scale_stride,
                            std::span<uint8_t> output_blocks, std::span<uint8_t> output_scales) {
        const auto before_storage = storage;
        const auto before_blocks = blocks;
        const auto before_scales = scales;
        check(!unpack_gguf_mxfp4_rows(input, blocks_per_row, block_stride, scale_stride, output_blocks, output_scales));
        check(storage == before_storage && blocks == before_blocks && scales == before_scales);
    };
    reject({}, 2, 32, 2, blocks, scales);
    reject(source.first(67), 2, 32, 2, blocks, scales);
    reject(source, 0, 32, 2, blocks, scales);
    reject(source, 3, 48, 3, blocks, scales);
    reject(source, 2, 31, 2, blocks, scales);
    reject(source, 2, 32, 1, blocks, scales);
    reject(source, 2, 32, 2, std::span<uint8_t>(blocks).first(63), scales);
    reject(source, 2, 32, 2, blocks, std::span<uint8_t>(scales).first(3));
    reject(source, 2, std::numeric_limits<size_t>::max(), 2, blocks, scales);
    reject(source, 2, 32, std::numeric_limits<size_t>::max(), blocks, scales);
    reject(source, 2, 32, 2, std::span<uint8_t>(storage).subspan(10, 64), scales);
    reject(source, 2, 32, 2, blocks, std::span<uint8_t>(storage).subspan(66, 4));
    reject(source, 2, 32, 2, std::span<uint8_t>(storage).subspan(80, 64), std::span<uint8_t>(storage).subspan(140, 4));
    reject(std::span<const uint8_t>(storage).subspan(63, 68), 2, 32, 2, std::span<uint8_t>(storage).first(64), scales);
    reject(source, 2, 32, 2, std::span<uint8_t>(storage).subspan(84, 64), std::span<uint8_t>(storage).subspan(83, 4));
}

void test_mxfp4_weight_storage_padding()
{
    // Three 32-column rows exercise uint32 scale padding and segment alignment.
    std::array<uint8_t, 48> blocks;
    for (size_t index = 0; index < blocks.size(); ++index) blocks[index] = static_cast<uint8_t>(index * 11 + 1);
    const std::array<uint8_t, 3> scales = {125, 127, 129};
    const std::array<float, 3> float_bias = {0.25f, -0.5f, 1.5f};
    const std::array<uint16_t, 3> bfloat16_bias = {0x3e80, 0xbf00, 0x3fc0};
    for (size_t alignment : {size_t{4}, size_t{64}, size_t{256}})
    {
        const size_t packed_segment_size = (blocks.size() + alignment - 1) / alignment * alignment;
        const size_t scales_segment_size = (scales.size() + alignment - 1) / alignment * alignment;
        const size_t bias_segment_size = (float_bias.size() * sizeof(float) + alignment - 1) / alignment * alignment;
        const size_t storage_size = packed_segment_size + scales_segment_size + bias_segment_size;
        constexpr size_t guard_size = 16;
        std::vector<uint32_t> allocation((storage_size + guard_size * 2) / sizeof(uint32_t), 0xa5a5a5a5);
        uint8_t* base = reinterpret_cast<uint8_t*>(allocation.data());
        std::span<uint8_t> storage(base + guard_size, storage_size);
        for (int bias_mode = 0; bias_mode < 3; ++bias_mode)
        {
            // Reusing a poisoned buffer ensures absent bias never inherits prior data.
            std::fill(storage.begin(), storage.end(), 0xa5);
            pack_mxfp4_weight_storage(storage, packed_segment_size, scales_segment_size, blocks, scales,
                                      bias_mode == 1 ? std::span<const float>(float_bias) : std::span<const float>(),
                                      bias_mode == 2 ? std::span<const uint16_t>(bfloat16_bias) : std::span<const uint16_t>());
            std::vector<uint8_t> expected(storage_size, 0);
            std::copy(blocks.begin(), blocks.end(), expected.begin());
            std::copy(scales.begin(), scales.end(), expected.begin() + packed_segment_size);
            if (bias_mode != 0)
                std::memcpy(expected.data() + packed_segment_size + scales_segment_size, float_bias.data(), float_bias.size() * sizeof(float));
            check(std::equal(storage.begin(), storage.end(), expected.begin()));
            check(std::all_of(base, base + guard_size, [](uint8_t value) { return value == 0xa5; }));
            check(std::all_of(storage.data() + storage_size, storage.data() + storage_size + guard_size, [](uint8_t value) { return value == 0xa5; }));
        }
    }
}

void test_weightstore_finalize()
{
    WeightStore weights;
    TensorData tensor;
    tensor.dtype = DType::Float32;
    tensor.shape = {1};
    tensor.float32_data = {3.5f};
    auto added = weights.add("weight", std::move(tensor));
    check(static_cast<bool>(added));
    const TensorHandle handle = added.value();

    check(weights.find_handle("weight") == handle);
    weights.finalize();
    check(weights.find_handle("weight") == invalid_tensor_handle);
    check(weights.size() == 1);
    check(weights.at(handle).float32_data == std::vector<float>{3.5f});

    TensorData later_tensor;
    later_tensor.dtype = DType::Float32;
    later_tensor.shape = {1};
    later_tensor.float32_data = {7.0f};
    auto later = weights.add("later", std::move(later_tensor));
    check(static_cast<bool>(later));
    check(weights.find_handle("weight") == invalid_tensor_handle);
    check(weights.find_handle("later") == later.value());
    check(weights.at(later.value()).float32_data == std::vector<float>{7.0f});
    check(!weights.add("later", TensorData{}));
}

void test_mxfp4_cpu_kernel_and_fused_gate_up()
{
    const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
    TensorData matrix;
    matrix.dtype = DType::MxFp4;
    matrix.shape = {4, 32};
    matrix.mxfp4_scales = {127, 127, 127, 127};
    matrix.mxfp4_blocks.resize(64);
    for (size_t row = 0; row < 4; ++row)
    {
        for (size_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t low = static_cast<uint8_t>((byte + row) % 16);
            const uint8_t high = static_cast<uint8_t>((byte * 3 + row + 1) % 16);
            matrix.mxfp4_blocks[row * 16 + byte] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    ActivationBuffer input(2, 32);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>(column % 7) - 3) * (row == 0 ? 0.25f : -0.125f);
    }
    static constexpr float values[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    auto scalar_row = [&](size_t matrix_row, size_t input_row) {
        float sum = 0.0f;
        for (size_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t packed = matrix.mxfp4_blocks[matrix_row * 16 + byte];
            sum += values[packed & 0x0f] * input.row(input_row)[byte * 2];
            sum += values[packed >> 4] * input.row(input_row)[byte * 2 + 1];
        }
        return sum;
    };

    const ActivationBuffer projected = forward_linear(matrix,
                                                      input,
                                                      g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
    {
        for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
        {
            check_near(projected.row(input_row)[matrix_row], scalar_row(matrix_row, input_row), 1e-5f);
        }
    }
    const uint64_t q8_flags = (g_test_optimization_flags | OptimizationCpuMxfp4Q8)
                              & ~OptimizationCpuPackedWeights;
    check(!has_flag(q8_flags, OptimizationCpuPackedWeights));
    const ActivationBuffer unpacked_q8_projected = forward_linear(matrix,
                                                                  input,
                                                                  q8_flags);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
        for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
            check_near(unpacked_q8_projected.row(input_row)[matrix_row], scalar_row(matrix_row, input_row), 0.15f);
    const ActivationBuffer packed_q8_projected = forward_linear(matrix,
                                                                input,
                                                                q8_flags | OptimizationCpuPackedWeights);
    CompiledOperator packed_owner;
    ActivationBuffer owner_unpacked_q8_projected;
    forward_linear(matrix,
                   input,
                   owner_unpacked_q8_projected,
                   q8_flags,
                   &packed_owner,
                   ExecutionBackend::Cpu);
    check(static_cast<bool>(!packed_owner.mxfp4_q8_packed));
    const ActivationBuffer owner_packed_q8_projected = forward_linear(matrix,
                                                                      input,
                                                                      q8_flags | OptimizationCpuPackedWeights,
                                                                      &packed_owner,
                                                                      ExecutionBackend::Cpu);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
        for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
        {
            check_near(packed_q8_projected.row(input_row)[matrix_row], scalar_row(matrix_row, input_row), 0.15f);
            check_near(packed_q8_projected.row(input_row)[matrix_row], unpacked_q8_projected.row(input_row)[matrix_row], 0.15f);
            check_near(owner_packed_q8_projected.row(input_row)[matrix_row], packed_q8_projected.row(input_row)[matrix_row], 0.15f);
        }
    if (mxfp4_q8_ready())
        check(static_cast<bool>(packed_owner.mxfp4_q8_packed));
    const std::shared_ptr<const Mxfp4Q8PackedMatrix> packed_owner_sidecar = packed_owner.mxfp4_q8_packed;
    ActivationBuffer owner_packed_q8_projected_again;
    forward_linear(matrix,
                   input,
                   owner_packed_q8_projected_again,
                   q8_flags | OptimizationCpuPackedWeights,
                   &packed_owner,
                   ExecutionBackend::Cpu);
    check(static_cast<bool>(packed_owner.mxfp4_q8_packed == packed_owner_sidecar));
    auto vulkan_projection = Mxfp4Linear_vulkan::create(matrix,
                                                        nullptr,
                                                        automatic_vulkan_device_index,
                                                        vulkan_runtime,
                                                        g_test_optimization_flags);
    if (get_gpu_count() > 0)
    {
        check(static_cast<bool>(vulkan_projection));
        check(static_cast<bool>(vulkan_projection->input_columns() == 32));
        check(static_cast<bool>(vulkan_projection->output_columns() == 4));
        ActivationBuffer vulkan_output;
        check(static_cast<bool>(vulkan_projection->forward(input, vulkan_output)));
        for (size_t input_row = 0; input_row < input.rows(); ++input_row)
        {
            for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
            {
                check_near(vulkan_output.row(input_row)[matrix_row], projected.row(input_row)[matrix_row], 1e-4f);
            }
        }
        ActivationBuffer four_row_input(4, 32);
        for (size_t row = 0; row < four_row_input.rows(); ++row)
        {
            for (uint32_t column = 0; column < four_row_input.columns(); ++column)
            {
                four_row_input.row(row)[column] = static_cast<float>(static_cast<int>((column + row * 5) % 11) - 5) * 0.0625f;
            }
        }
        const ActivationBuffer four_row_cpu = forward_linear(matrix,
                                                             four_row_input,
                                                             g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        ActivationBuffer four_row_vulkan;
        check(static_cast<bool>(vulkan_projection->forward(four_row_input, four_row_vulkan)));
        for (size_t row = 0; row < four_row_input.rows(); ++row)
        {
            for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
            {
                check_near(four_row_vulkan.row(row)[matrix_row], four_row_cpu.row(row)[matrix_row], 1e-4f);
            }
        }
    }
    else
    {
        check(static_cast<bool>(!vulkan_projection));
    }

    ActivationBuffer decode_input(1, 32);
    std::copy_n(input.row(0), input.columns(), decode_input.row(0));
    const ActivationBuffer decoded = forward_linear(matrix,
                                                    decode_input,
                                                    g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
    for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
    {
        check_near(decoded.row(0)[matrix_row], scalar_row(matrix_row, 0), 1e-5f);
    }
    TensorData odd_matrix = matrix;
    odd_matrix.shape[0] = 3;
    odd_matrix.mxfp4_blocks.resize(3 * 16);
    odd_matrix.mxfp4_scales.resize(3);
    const ActivationBuffer odd_projected = forward_linear(odd_matrix,
                                                          input,
                                                          g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
    check(static_cast<bool>(odd_projected.columns() == 3));
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
    {
        for (size_t matrix_row = 0; matrix_row < 3; ++matrix_row)
        {
            check_near(odd_projected.row(input_row)[matrix_row], scalar_row(matrix_row, input_row), 1e-5f);
        }
    }
    auto odd_vulkan_projection = Mxfp4Linear_vulkan::create(odd_matrix,
                                                            nullptr,
                                                            automatic_vulkan_device_index,
                                                            vulkan_runtime,
                                                            g_test_optimization_flags);
    if (get_gpu_count() > 0)
    {
        check(static_cast<bool>(odd_vulkan_projection));
        ActivationBuffer odd_vulkan_output;
        check(static_cast<bool>(odd_vulkan_projection->forward(input, odd_vulkan_output)));
        for (size_t input_row = 0; input_row < input.rows(); ++input_row)
        {
            for (size_t matrix_row = 0; matrix_row < 3; ++matrix_row)
            {
                check_near(odd_vulkan_output.row(input_row)[matrix_row], odd_projected.row(input_row)[matrix_row], 1e-4f);
            }
        }
    }
    else
    {
        check(static_cast<bool>(!odd_vulkan_projection));
    }

    TensorData bias;
    bias.dtype = DType::Float32;
    bias.shape = {4};
    bias.float32_data = {0.25f, -0.5f, 0.75f, -1.0f};
    CompiledOperator biased_owner;
    ActivationBuffer biased_owner_unpacked;
    forward_linear(matrix,
                   bias,
                   input,
                   biased_owner_unpacked,
                   q8_flags,
                   &biased_owner,
                   ExecutionBackend::Cpu);
    check(static_cast<bool>(!biased_owner.mxfp4_q8_packed));
    const ActivationBuffer biased_owner_output = forward_linear(matrix,
                                                                bias,
                                                                input,
                                                                q8_flags | OptimizationCpuPackedWeights,
                                                                &biased_owner,
                                                                ExecutionBackend::Cpu);
    const ActivationBuffer biased_reference = forward_linear(matrix,
                                                             bias,
                                                             input,
                                                             q8_flags | OptimizationCpuPackedWeights);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
        for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
            check_near(biased_owner_output.row(input_row)[matrix_row],
                       biased_reference.row(input_row)[matrix_row],
                       0.15f);
    if (mxfp4_q8_ready())
        check(static_cast<bool>(biased_owner.mxfp4_q8_packed));
    const std::shared_ptr<const Mxfp4Q8PackedMatrix> biased_owner_sidecar = biased_owner.mxfp4_q8_packed;
    ActivationBuffer biased_owner_output_again;
    forward_linear(matrix,
                   bias,
                   input,
                   biased_owner_output_again,
                   q8_flags | OptimizationCpuPackedWeights,
                   &biased_owner,
                   ExecutionBackend::Cpu);
    check(static_cast<bool>(biased_owner.mxfp4_q8_packed == biased_owner_sidecar));
    for (bool with_bias : {false, true})
    {
        CompiledOperator missing_gpu_product;
        ActivationBuffer unavailable_output;
        bool rejected = false;
        try
        {
            if (with_bias)
                forward_linear(matrix, bias, input, unavailable_output, q8_flags, &missing_gpu_product, ExecutionBackend::Vulkan);
            else
                forward_linear(matrix, input, unavailable_output, q8_flags, &missing_gpu_product, ExecutionBackend::Vulkan);
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        check(rejected);
        check(!missing_gpu_product.mxfp4_q8_packed);
    }
    ActivationBuffer fused;
    forward_gate_up_mxfp4(matrix,
                          &bias,
                          input,
                          ExpertActivation::GptOssSwiGlu,
                          7.0f,
                          fused,
                          g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
    check(static_cast<bool>(fused.rows() == input.rows()));
    check(static_cast<bool>(fused.columns() == 2));
    for (float sigmoid_scale : {1.0f, 1.702f})
    {
        for (int step = -1000; step <= 700; ++step)
        {
            const float value = static_cast<float>(step) * 0.01f;
            const float expected = value / (1.0f + std::exp(-sigmoid_scale * value));
            check_near(float_silu(value, sigmoid_scale), expected, 1e-5f);
            check_near(scaled_silu(value,
                                   sigmoid_scale,
                                   g_test_optimization_flags),
                       expected,
                       1e-5f);
        }
    }
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
    {
        for (size_t column = 0; column < fused.columns(); ++column)
        {
            const float gate = std::min(scalar_row(column * 2, input_row) + bias.float32_data[column * 2], 7.0f);
            const float linear = std::clamp(scalar_row(column * 2 + 1, input_row) + bias.float32_data[column * 2 + 1], -7.0f, 7.0f);
            const float expected = gate / (1.0f + std::exp(-1.702f * gate)) * (linear + 1.0f);
            check_near(fused.row(input_row)[column], expected, 1e-5f);
        }
    }

    TensorData expert_gate_up;
    expert_gate_up.dtype = DType::MxFp4;
    expert_gate_up.shape = {64, 32};
    expert_gate_up.mxfp4_scales.resize(64);
    expert_gate_up.mxfp4_blocks.resize(64 * 16);
    for (size_t row = 0; row < 64; ++row)
    {
        expert_gate_up.mxfp4_scales[row] = static_cast<uint8_t>(125 + row % 5);
        for (size_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t low = static_cast<uint8_t>((row * 3 + byte * 5 + 1) % 16);
            const uint8_t high = static_cast<uint8_t>((row * 7 + byte * 2 + 4) % 16);
            expert_gate_up.mxfp4_blocks[row * 16 + byte] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    TensorData expert_gate_up_bias;
    expert_gate_up_bias.dtype = DType::Float32;
    expert_gate_up_bias.shape = {64};
    expert_gate_up_bias.float32_data.resize(64);
    for (size_t row = 0; row < 64; ++row)
    {
        expert_gate_up_bias.float32_data[row] = static_cast<float>(static_cast<int>(row % 9) - 4) * 0.03125f;
    }
    TensorData expert_down;
    expert_down.dtype = DType::MxFp4;
    expert_down.shape = {7, 32};
    expert_down.mxfp4_scales.resize(7);
    expert_down.mxfp4_blocks.resize(7 * 16);
    for (size_t row = 0; row < 7; ++row)
    {
        expert_down.mxfp4_scales[row] = static_cast<uint8_t>(126 + row % 3);
        for (size_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t low = static_cast<uint8_t>((row * 11 + byte * 3 + 2) % 16);
            const uint8_t high = static_cast<uint8_t>((row * 5 + byte * 7 + 8) % 16);
            expert_down.mxfp4_blocks[row * 16 + byte] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    TensorData expert_down_bias;
    expert_down_bias.dtype = DType::Float32;
    expert_down_bias.shape = {7};
    expert_down_bias.float32_data.resize(7);
    for (size_t row = 0; row < 7; ++row)
    {
        expert_down_bias.float32_data[row] = static_cast<float>(static_cast<int>(row) - 3) * 0.0625f;
    }
    constexpr float expert_activation_limit = 5.25f;
    ActivationBuffer repeated_expert_input(4, 32);
    for (uint32_t column = 0;
         column < repeated_expert_input.columns();
         ++column)
    {
        repeated_expert_input.row(0)[column] = static_cast<float>(static_cast<int>(column % 11) - 5)
                                               * 0.03125f;
        repeated_expert_input.row(1)[column] = static_cast<float>(static_cast<int>((column * 3) % 17) - 8)
                                               * 0.015625f;
        repeated_expert_input.row(3)[column] = static_cast<float>(static_cast<int>((column * 5) % 19) - 9)
                                               * 0.015625f;
    }
    std::copy_n(repeated_expert_input.row(0),
                repeated_expert_input.columns(),
                repeated_expert_input.row(2));
    ActivationBuffer linear_scratch;
    ActivationBuffer repeated_activated;
    forward_gate_up_mxfp4(expert_gate_up,
                          &expert_gate_up_bias,
                          repeated_expert_input,
                          ExpertActivation::DeepSeekSwiGlu,
                          expert_activation_limit,
                          repeated_activated,
                          g_test_optimization_flags,
                          &linear_scratch);
    const std::byte* linear_scratch_data = linear_scratch.bytes().data();
    const uint64_t linear_scratch_capacity = linear_scratch.allocated_bytes();
    ActivationBuffer resized_activated;
    forward_gate_up_mxfp4(expert_gate_up,
                          &expert_gate_up_bias,
                          input,
                          ExpertActivation::DeepSeekSwiGlu,
                          expert_activation_limit,
                          resized_activated,
                          g_test_optimization_flags,
                          &linear_scratch);
    ActivationBuffer resized_reference;
    forward_gate_up_mxfp4(expert_gate_up,
                          &expert_gate_up_bias,
                          input,
                          ExpertActivation::DeepSeekSwiGlu,
                          expert_activation_limit,
                          resized_reference,
                          g_test_optimization_flags);
    check(resized_activated.rows() == resized_reference.rows());
    check(resized_activated.columns() == resized_reference.columns());
    check(linear_scratch.bytes().data() == linear_scratch_data);
    check(linear_scratch.allocated_bytes() == linear_scratch_capacity);
    for (size_t index = 0; index < resized_reference.values().size(); ++index)
        check_near(resized_activated.values()[index], resized_reference.values()[index], 1e-5f);
    const ActivationBuffer repeated_reference = forward_linear(expert_down,
                                                               expert_down_bias,
                                                               repeated_activated,
                                                               g_test_optimization_flags);
    ActivationBuffer repeated_output;
    Mxfp4Task repeated_task;
    repeated_task.gate_up = &expert_gate_up;
    repeated_task.gate_up_bias = &expert_gate_up_bias;
    repeated_task.down = &expert_down;
    repeated_task.down_bias = &expert_down_bias;
    repeated_task.input = &repeated_expert_input;
    repeated_task.output = &repeated_output;
    repeated_task.activation = ExpertActivation::DeepSeekSwiGlu;
    repeated_task.activation_limit = expert_activation_limit;

    ActivationBuffer single_expert_input(1, 32);
    std::copy_n(repeated_expert_input.row(0), single_expert_input.columns(), single_expert_input.row(0));
    ActivationBuffer single_token_scratch;
    ActivationBuffer single_token_activated;
    forward_gate_up_mxfp4(expert_gate_up,
                          &expert_gate_up_bias,
                          single_expert_input,
                          ExpertActivation::DeepSeekSwiGlu,
                          expert_activation_limit,
                          single_token_activated,
                          g_test_optimization_flags,
                          &single_token_scratch);
    check(single_token_scratch.allocated_bytes() == 0);
    auto check_mxfp4_rejected = [&](Mxfp4Task task, const ActivationBuffer& task_input) {
        ActivationBuffer output(1, expert_down.shape[0]);
        std::fill_n(output.row(0), output.columns(), 17.0f);
        task.input = &task_input;
        task.output = &output;
        Mxfp4Scratch scratch;
        check(!forward_experts_mxfp4(std::span<const Mxfp4Task>(&task, 1), &scratch, g_test_optimization_flags));
        check(scratch.activated.empty());
        check(scratch.linear.empty());
        check(scratch.effective_tasks.empty());
        check(std::all_of(output.values().begin(), output.values().end(), [](float value) { return value == 17.0f; }));
    };
    TensorData bfloat16_gate_up = expert_gate_up;
    bfloat16_gate_up.dtype = DType::BFloat16;
    Mxfp4Task bfloat16_task = repeated_task;
    bfloat16_task.gate_up = &bfloat16_gate_up;
    check_mxfp4_rejected(bfloat16_task, single_expert_input);
    check_mxfp4_rejected(bfloat16_task, repeated_expert_input);
    Mxfp4Task null_gate_task = repeated_task;
    null_gate_task.gate_up = nullptr;
    check_mxfp4_rejected(null_gate_task, single_expert_input);
    TensorData malformed_gate_up = expert_gate_up;
    malformed_gate_up.shape[0] -= 1;
    Mxfp4Task malformed_task = repeated_task;
    malformed_task.gate_up = &malformed_gate_up;
    check_mxfp4_rejected(malformed_task, single_expert_input);
    Mxfp4Task relu_task = repeated_task;
    relu_task.activation = ExpertActivation::Relu;
    check_mxfp4_rejected(relu_task, single_expert_input);
    check_mxfp4_rejected(relu_task, repeated_expert_input);

    Mxfp4Task valid_task = repeated_task;
    ActivationBuffer valid_output;
    valid_task.input = &single_expert_input;
    valid_task.output = &valid_output;
    Mxfp4Scratch valid_scratch;
    check(forward_experts_mxfp4(std::span<const Mxfp4Task>(&valid_task, 1), &valid_scratch, 0));
    ActivationBuffer valid_reference_activated;
    forward_gate_up_mxfp4(expert_gate_up,
                          &expert_gate_up_bias,
                          single_expert_input,
                          ExpertActivation::DeepSeekSwiGlu,
                          expert_activation_limit,
                          valid_reference_activated,
                          0);
    const ActivationBuffer valid_reference = forward_linear(expert_down,
                                                            expert_down_bias,
                                                            valid_reference_activated,
                                                            0);
    for (size_t index = 0; index < valid_output.values().size(); ++index)
        check_near(valid_output.values()[index], valid_reference.values()[index], 1e-5f);

    Mxfp4Task direct_task = valid_task;
    direct_task.activation = ExpertActivation::Silu;
    ActivationBuffer direct_output;
    direct_task.output = &direct_output;
    Mxfp4Scratch direct_scratch;
    check(forward_experts_mxfp4(std::span<const Mxfp4Task>(&direct_task, 1), &direct_scratch, 0));
    ActivationBuffer direct_reference_activated;
    forward_gate_up_mxfp4(expert_gate_up,
                          &expert_gate_up_bias,
                          single_expert_input,
                          ExpertActivation::Silu,
                          expert_activation_limit,
                          direct_reference_activated,
                          0);
    const ActivationBuffer direct_reference = forward_linear(expert_down,
                                                             expert_down_bias,
                                                             direct_reference_activated,
                                                             0);
    for (size_t index = 0; index < direct_output.values().size(); ++index)
        check_near(direct_output.values()[index], direct_reference.values()[index], 1e-5f);

    const uint64_t fallback_scalar_flags = g_test_optimization_flags & ~OptimizationCpuMxfp4Q8;
    const uint64_t fallback_q8_flags = g_test_optimization_flags | OptimizationCpuMxfp4Q8;
    ActivationBuffer reused_fused_output;
    forward_gate_up_mxfp4(expert_gate_up,
                          &expert_gate_up_bias,
                          repeated_expert_input,
                          ExpertActivation::DeepSeekSwiGlu,
                          expert_activation_limit,
                          reused_fused_output,
                          fallback_scalar_flags);
    ActivationBuffer scalar_fused_reference;
    forward_gate_up_mxfp4(expert_gate_up,
                          &expert_gate_up_bias,
                          repeated_expert_input,
                          ExpertActivation::DeepSeekSwiGlu,
                          expert_activation_limit,
                          scalar_fused_reference,
                          fallback_scalar_flags);
    const std::byte* reused_fused_storage = reused_fused_output.bytes().data();
    const uint64_t reused_fused_capacity = reused_fused_output.allocated_bytes();
    check(static_cast<bool>(reused_fused_output.rows() == repeated_expert_input.rows()));
    check(static_cast<bool>(reused_fused_output.columns() == expert_gate_up.shape[0] / 2));
    check(static_cast<bool>(reused_fused_capacity >= reused_fused_output.bytes().size()));
    for (size_t index = 0; index < scalar_fused_reference.values().size(); ++index)
        check_near(reused_fused_output.values()[index], scalar_fused_reference.values()[index], 1e-5f);

    forward_gate_up_mxfp4(expert_gate_up,
                          nullptr,
                          single_expert_input,
                          ExpertActivation::Silu,
                          0.0f,
                          reused_fused_output,
                          fallback_q8_flags);
    ActivationBuffer q8_single_reference;
    forward_gate_up_mxfp4(expert_gate_up,
                          nullptr,
                          single_expert_input,
                          ExpertActivation::Silu,
                          0.0f,
                          q8_single_reference,
                          fallback_q8_flags);
    check(static_cast<bool>(reused_fused_output.rows() == single_expert_input.rows()));
    check(static_cast<bool>(reused_fused_output.columns() == q8_single_reference.columns()));
    check(static_cast<bool>(reused_fused_output.bytes().data() == reused_fused_storage));
    check(static_cast<bool>(reused_fused_output.allocated_bytes() == reused_fused_capacity));
    for (size_t index = 0; index < q8_single_reference.values().size(); ++index)
        check_near(reused_fused_output.values()[index], q8_single_reference.values()[index], 1e-5f);

    forward_gate_up_mxfp4(expert_gate_up,
                          nullptr,
                          repeated_expert_input,
                          ExpertActivation::Silu,
                          0.0f,
                          reused_fused_output,
                          fallback_q8_flags);
    ActivationBuffer q8_repeated_reference;
    forward_gate_up_mxfp4(expert_gate_up,
                          nullptr,
                          repeated_expert_input,
                          ExpertActivation::Silu,
                          0.0f,
                          q8_repeated_reference,
                          fallback_q8_flags);
    check(static_cast<bool>(reused_fused_output.rows() == repeated_expert_input.rows()));
    check(static_cast<bool>(reused_fused_output.bytes().data() == reused_fused_storage));
    check(static_cast<bool>(reused_fused_output.allocated_bytes() == reused_fused_capacity));
    for (size_t index = 0; index < q8_repeated_reference.values().size(); ++index)
        check_near(reused_fused_output.values()[index], q8_repeated_reference.values()[index], 1e-5f);

    Mxfp4Scratch reused_scratch;
    ActivationBuffer q8_output;
    ActivationBuffer scalar_output;
    ActivationBuffer q8_again_output;
    Mxfp4Task reused_task = valid_task;
    reused_task.output = &q8_output;
    const uint64_t q8_reuse_flags = OptimizationCpuMxfp4Q8 | OptimizationCpuPackedWeights;
    check(forward_experts_mxfp4(std::span<const Mxfp4Task>(&reused_task, 1), &reused_scratch, q8_reuse_flags));
    const std::byte* reused_projection_storage = reused_scratch.activated[0].bytes().data();
    for (const auto& packed : reused_scratch.q8_down_packed)
        check(!packed);
    reused_task.output = &scalar_output;
    check(forward_experts_mxfp4(std::span<const Mxfp4Task>(&reused_task, 1), &reused_scratch, 0));
    for (const auto& packed : reused_scratch.q8_down_packed)
        check(!packed);
    reused_task.output = &q8_again_output;
    check(forward_experts_mxfp4(std::span<const Mxfp4Task>(&reused_task, 1), &reused_scratch, q8_reuse_flags));
    check(reused_scratch.activated[0].bytes().data() == reused_projection_storage);
    for (const auto& packed : reused_scratch.q8_down_packed)
        check(!packed);
    for (size_t index = 0; index < scalar_output.values().size(); ++index)
        check_near(scalar_output.values()[index], valid_reference.values()[index], 1e-5f);
    for (size_t index = 0; index < q8_output.values().size(); ++index)
    {
        check_near(q8_output.values()[index], q8_again_output.values()[index], 1e-5f);
    }
    Mxfp4Scratch repeated_scratch;
    check(static_cast<bool>(forward_experts_mxfp4(std::span<const Mxfp4Task>(&repeated_task,
                                                                             1),
                                                  &repeated_scratch,
                                                  g_test_optimization_flags)));
    check(static_cast<bool>(repeated_scratch.physical_input_rows
                            == std::vector<uint32_t>({3})));
    check(static_cast<bool>(repeated_output.rows()
                            == repeated_reference.rows()));
    for (size_t row = 0;
         row < repeated_output.rows();
         ++row)
    {
        for (uint32_t column = 0;
             column < repeated_output.columns();
             ++column)
        {
            check_near(repeated_output.row(row)[column],
                       repeated_reference.row(row)[column],
                       1e-5f);
        }
    }
    const uint64_t q8_expert_flags = g_test_optimization_flags | OptimizationCpuMxfp4Q8;
    ActivationBuffer q8_repeated_output;
    Mxfp4Scratch q8_repeated_scratch;
    repeated_task.output = &q8_repeated_output;
    check(static_cast<bool>(forward_experts_mxfp4(std::span<const Mxfp4Task>(&repeated_task, 1),
                                                  &q8_repeated_scratch,
                                                  q8_expert_flags)));
    check(q8_repeated_output.rows() == repeated_reference.rows());
    check(q8_repeated_output.columns() == repeated_reference.columns());
    for (size_t row = 0; row < q8_repeated_output.rows(); ++row)
    {
        for (uint32_t column = 0;
             column < q8_repeated_output.columns();
             ++column)
        {
            check_near(q8_repeated_output.row(row)[column],
                       repeated_reference.row(row)[column],
                       2.0f);
        }
    }
    {
        // Retain inner buffers across changing ready batches and decode/prefill.
        ActivationBuffer single_input(1, repeated_expert_input.columns());
        std::copy_n(repeated_expert_input.row(0), single_input.columns(), single_input.row(0));
        std::array<ActivationBuffer, 2> outputs;
        std::array<Mxfp4Task, 2> tasks = {repeated_task, repeated_task};
        Mxfp4Scratch scratch;
        std::array<const std::byte*, 2> activated = {};
        std::array<const std::byte*, 2> linear = {};
        std::array<const std::byte*, 2> unique_inputs = {};
        std::array<const std::byte*, 2> unique_outputs = {};
        std::array<const uint32_t*, 2> row_maps = {};
        std::array<const int8_t*, 2> q8_activated = {};
        for (size_t pass = 0; pass < 8; ++pass)
        {
            const size_t count = pass == 1 || pass == 5 ? 1 : 2;
            for (size_t index = 0; index < tasks.size(); ++index)
            {
                tasks[index].input = pass == 3 || pass == 5 || pass == 6 ? &single_input : &repeated_expert_input;
                tasks[index].output = &outputs[index];
            }
            check(forward_experts_mxfp4(std::span<const Mxfp4Task>(tasks.data(), count), &scratch, q8_expert_flags));
            check(scratch.activated.size() == 2);
            check(scratch.linear.size() == 2);
            check(scratch.unique_input.size() == 2);
            check(scratch.unique_output.size() == 2);
            check(scratch.unique_row_maps.size() == 2);
            if (pass >= 3)
                check(scratch.packed_gate_up.size() == 2);
            for (size_t index = 0; index < 2; ++index)
            {
                if (pass == 0)
                {
                    activated[index] = scratch.activated[index].bytes().data();
                    linear[index] = scratch.linear[index].bytes().data();
                    unique_inputs[index] = scratch.unique_input[index].bytes().data();
                    unique_outputs[index] = scratch.unique_output[index].bytes().data();
                    row_maps[index] = scratch.unique_row_maps[index].data();
                    check(activated[index] && unique_inputs[index] && unique_outputs[index] && row_maps[index]);
                    if (!scratch.q8_activated.empty())
                        q8_activated[index] = scratch.q8_activated[index].values.data();
                }
                check(scratch.activated[index].bytes().data() == activated[index]);
                check(scratch.linear[index].bytes().data() == linear[index]);
                check(scratch.unique_input[index].bytes().data() == unique_inputs[index]);
                check(scratch.unique_output[index].bytes().data() == unique_outputs[index]);
                check(scratch.unique_row_maps[index].data() == row_maps[index]);
                if (q8_activated[index])
                {
                    check(scratch.q8_activated.size() == 2);
                    check(scratch.q8_activated[index].values.data() == q8_activated[index]);
                }
            }
            std::array<ActivationBuffer, 2> expected;
            std::array<Mxfp4Task, 2> reference = tasks;
            for (size_t index = 0; index < count; ++index)
                reference[index].output = &expected[index];
            check(forward_experts_mxfp4(std::span<const Mxfp4Task>(reference.data(), count), nullptr, q8_expert_flags));
            for (size_t index = 0; index < count; ++index)
                check(std::equal(outputs[index].values().begin(), outputs[index].values().end(), expected[index].values().begin(), expected[index].values().end()));
        }
    }
    if (mxfp4_q8_kernel_available())
    {
        // Reuse storage, not quantized values or owners from the previous call.
        ActivationBuffer reuse_input(2, repeated_expert_input.columns());
        for (size_t row = 0; row < reuse_input.rows(); ++row)
            std::copy_n(repeated_expert_input.row(row), reuse_input.columns(), reuse_input.row(row));
        ActivationBuffer other_input = reuse_input;
        other_input.row(1)[0] += 0.125f;
        ActivationBuffer single_input(1, reuse_input.columns());
        std::copy_n(reuse_input.row(0), single_input.columns(), single_input.row(0));
        std::array<ActivationBuffer, 2> reuse_outputs;
        std::array<Mxfp4Task, 2> reuse_tasks = {repeated_task, repeated_task};
        reuse_tasks[0].input = &reuse_input;
        reuse_tasks[1].input = &other_input;
        for (size_t index = 0; index < reuse_tasks.size(); ++index)
            reuse_tasks[index].output = &reuse_outputs[index];
        Mxfp4Scratch reuse_scratch;
        std::array<const int8_t*, 2> values = {};
        std::array<const float*, 2> scales = {};
        for (size_t pass = 0; pass < 7; ++pass)
        {
            reuse_input.row(1)[0] += 0.03125f;
            reuse_tasks[0].input = pass == 5 ? &single_input : &reuse_input;
            reuse_tasks[1].input = pass == 5 ? &single_input : pass == 1 ? &reuse_input
                                                                         : &other_input;
            const size_t task_count = pass == 3 ? 1 : 2;
            const uint64_t flags = pass == 2 ? q8_expert_flags & ~OptimizationCpuMxfp4Q8 : q8_expert_flags;
            const std::span<const Mxfp4Task> tasks(reuse_tasks.data(), task_count);
            check(forward_experts_mxfp4(tasks, &reuse_scratch, flags));
            if (pass == 0)
            {
                check(reuse_scratch.q8_inputs.size() == 2);
                for (size_t index = 0; index < 2; ++index)
                {
                    values[index] = reuse_scratch.q8_inputs[index].values.data();
                    scales[index] = reuse_scratch.q8_inputs[index].scales.data();
                    check(values[index] != nullptr && scales[index] != nullptr);
                }
            }
            for (size_t index = 0; index < 2; ++index)
            {
                check(reuse_scratch.q8_inputs[index].values.data() == values[index]);
                check(reuse_scratch.q8_inputs[index].scales.data() == scales[index]);
            }
            if (pass != 2 && (pass != 5 || mxfp4_kernel_kind() != MxFp4KernelKind::X86Avx512))
            {
                check(reuse_scratch.q8_input_owner[0] == 0);
                if (task_count == 2)
                    check(reuse_scratch.q8_input_owner[1] == (pass == 1 || pass == 5 ? 0 : 1));
            }
            std::array<ActivationBuffer, 2> fresh_outputs;
            std::array<Mxfp4Task, 2> fresh_tasks = reuse_tasks;
            for (size_t index = 0; index < task_count; ++index)
                fresh_tasks[index].output = &fresh_outputs[index];
            check(forward_experts_mxfp4(std::span<const Mxfp4Task>(fresh_tasks.data(), task_count), nullptr, flags));
            for (size_t index = 0; index < task_count; ++index)
                check(std::equal(reuse_outputs[index].values().begin(), reuse_outputs[index].values().end(), fresh_outputs[index].values().begin(), fresh_outputs[index].values().end()));
        }
    }
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512
        && mxfp4_q8_kernel_available())
    {
        ActivationBuffer grouping_single_input(1, repeated_expert_input.columns());
        for (uint32_t column = 0;
             column < grouping_single_input.columns();
             ++column)
        {
            const int pattern = static_cast<int>((column * 37 + 11) % 101) - 50;
            grouping_single_input.row(0)[column] = static_cast<float>(pattern) * 0.00073f
                                                   + static_cast<float>(column % 5) * 0.00019f;
        }
        ActivationBuffer grouping_distinct_input(2, grouping_single_input.columns());
        std::copy_n(grouping_single_input.row(0),
                    grouping_single_input.columns(),
                    grouping_distinct_input.row(0));
        for (uint32_t column = 0;
             column < grouping_distinct_input.columns();
             ++column)
        {
            const int pattern = static_cast<int>((column * 23 + 7) % 89) - 44;
            grouping_distinct_input.row(1)[column] = static_cast<float>(pattern) * 0.00061f
                                                     - static_cast<float>(column % 4) * 0.00013f;
        }
        ActivationBuffer grouping_repeated_input(2, grouping_single_input.columns());
        std::copy_n(grouping_single_input.row(0),
                    grouping_single_input.columns(),
                    grouping_repeated_input.row(0));
        std::copy_n(grouping_single_input.row(0),
                    grouping_single_input.columns(),
                    grouping_repeated_input.row(1));

        auto make_grouping_task = [&](const ActivationBuffer& task_input,
                                      ActivationBuffer& task_output) {
            Mxfp4Task task = repeated_task;
            task.input = &task_input;
            task.output = &task_output;
            return task;
        };
        ActivationBuffer grouping_single_output;
        Mxfp4Task grouping_single_task = make_grouping_task(grouping_single_input,
                                                            grouping_single_output);
        Mxfp4Scratch grouping_single_scratch;
        check(static_cast<bool>(forward_experts_mxfp4(std::span<const Mxfp4Task>(&grouping_single_task, 1),
                                                      &grouping_single_scratch,
                                                      q8_expert_flags)));

        ActivationBuffer grouping_distinct_single_output;
        ActivationBuffer grouping_distinct_multi_output;
        std::array<Mxfp4Task, 2> grouping_distinct_tasks = {
            make_grouping_task(grouping_single_input, grouping_distinct_single_output),
            make_grouping_task(grouping_distinct_input, grouping_distinct_multi_output)};
        Mxfp4Scratch grouping_distinct_scratch;
        check(static_cast<bool>(forward_experts_mxfp4(std::span<const Mxfp4Task>(grouping_distinct_tasks.data(),
                                                                                 grouping_distinct_tasks.size()),
                                                      &grouping_distinct_scratch,
                                                      q8_expert_flags)));

        ActivationBuffer grouping_repeated_single_output;
        ActivationBuffer grouping_repeated_multi_output;
        std::array<Mxfp4Task, 2> grouping_repeated_tasks = {
            make_grouping_task(grouping_single_input, grouping_repeated_single_output),
            make_grouping_task(grouping_repeated_input, grouping_repeated_multi_output)};
        Mxfp4Scratch grouping_repeated_scratch;
        check(static_cast<bool>(forward_experts_mxfp4(std::span<const Mxfp4Task>(grouping_repeated_tasks.data(),
                                                                                 grouping_repeated_tasks.size()),
                                                      &grouping_repeated_scratch,
                                                      q8_expert_flags)));
        check(static_cast<bool>(grouping_repeated_scratch.physical_input_rows
                                == std::vector<uint32_t>({1, 1})));

        auto check_grouping_output = [&](const ActivationBuffer& mixed_output) {
            check(mixed_output.rows() == grouping_single_output.rows());
            check(mixed_output.columns() == grouping_single_output.columns());
            for (uint32_t column = 0;
                 column < grouping_single_output.columns();
                 ++column)
            {
                const float expected = grouping_single_output.row(0)[column];
                const float tolerance = 1e-5f + 1e-5f * std::abs(expected);
                check(std::abs(mixed_output.row(0)[column] - expected) <= tolerance);
            }
        };
        check_grouping_output(grouping_distinct_single_output);
        check_grouping_output(grouping_repeated_single_output);
    }
    {
        // Changing the worker partition must not change SIMD activation tails.
        auto make_activation_weight = [](uint32_t rows, uint32_t columns) {
            TensorData tensor;
            tensor.dtype = DType::MxFp4;
            tensor.shape = {rows, columns};
            tensor.mxfp4_blocks.resize(static_cast<size_t>(rows) * columns / 2);
            tensor.mxfp4_scales.resize(static_cast<size_t>(rows) * columns / 32);
            std::fill_n(tensor.mxfp4_scales.data(), tensor.mxfp4_scales.size(), uint8_t{122});
            for (size_t index = 0; index < tensor.mxfp4_blocks.size(); ++index)
                tensor.mxfp4_blocks[index] = static_cast<uint8_t>((index * 37 + index / 23) % 256);
            return tensor;
        };
        TensorData activation_gate = make_activation_weight(256, 1024);
        TensorData activation_down = make_activation_weight(32, 128);
        ActivationBuffer activation_input(2, 1024);
        for (size_t row = 0; row < activation_input.rows(); ++row)
        {
            for (uint32_t column = 0; column < activation_input.columns(); ++column)
                activation_input.row(row)[column] = static_cast<float>(static_cast<int>((column * 31 + row * 17) % 103) - 51) * 0.0017f;
        }
        ActivationBuffer activation_reference;
        ActivationBuffer activation_output;
        Mxfp4Task activation_task;
        activation_task.gate_up = &activation_gate;
        activation_task.down = &activation_down;
        activation_task.input = &activation_input;
        activation_task.output = &activation_output;
        activation_task.activation = ExpertActivation::Silu;
        Mxfp4Scratch activation_scratch;
        CpuOpenMpThreadLimitScope thread_limit;
        const uint64_t activation_flags = (g_test_optimization_flags | OptimizationCpuFastSilu) & ~OptimizationCpuMxfp4Q8;
        for (uint32_t threads : {1u, 2u, 3u, 4u})
        {
            thread_limit.set(threads);
            check(forward_experts_mxfp4(std::span<const Mxfp4Task>(&activation_task, 1), &activation_scratch, activation_flags));
            if (threads == 1)
            {
                activation_reference = activation_output;
                continue;
            }
            check(activation_output.rows() == activation_reference.rows());
            check(activation_output.columns() == activation_reference.columns());
            check(std::memcmp(activation_output.bytes().data(), activation_reference.bytes().data(), activation_output.bytes().size()) == 0);
        }
    }
    for (size_t token_count : {size_t(1), repeated_expert_input.rows()})
    {
        ActivationBuffer silu_input(token_count, repeated_expert_input.columns());
        for (size_t row = 0; row < token_count; ++row)
        {
            std::copy_n(repeated_expert_input.row(row),
                        repeated_expert_input.columns(),
                        silu_input.row(row));
        }
        ActivationBuffer silu_activated;
        forward_gate_up_mxfp4(expert_gate_up,
                              nullptr,
                              silu_input,
                              ExpertActivation::Silu,
                              0.0f,
                              silu_activated,
                              g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const ActivationBuffer silu_reference = forward_linear(expert_down,
                                                               silu_activated,
                                                               g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        ActivationBuffer silu_output;
        Mxfp4Task silu_task;
        silu_task.gate_up = &expert_gate_up;
        silu_task.down = &expert_down;
        silu_task.input = &silu_input;
        silu_task.output = &silu_output;
        silu_task.activation = ExpertActivation::Silu;
        Mxfp4Scratch silu_scratch;
        check(static_cast<bool>(forward_experts_mxfp4(std::span<const Mxfp4Task>(&silu_task, 1),
                                                      &silu_scratch,
                                                      g_test_optimization_flags & ~OptimizationCpuMxfp4Q8)));
        check(silu_output.rows() == silu_reference.rows());
        check(silu_output.columns() == silu_reference.columns());
        for (size_t row = 0; row < silu_output.rows(); ++row)
        {
            for (uint32_t column = 0; column < silu_output.columns(); ++column)
            {
                check_near(silu_output.row(row)[column],
                           silu_reference.row(row)[column],
                           1e-5f);
            }
        }
    }
    repeated_expert_input.row(2)[0] += 0.03125f;
    check(static_cast<bool>(forward_experts_mxfp4(std::span<const Mxfp4Task>(&repeated_task,
                                                                             1),
                                                  &repeated_scratch,
                                                  g_test_optimization_flags)));
    check(static_cast<bool>(repeated_scratch.physical_input_rows
                            == std::vector<uint32_t>({4})));

    {
        // A mixed MXFP4/BF16 batch must reject the grouped MXFP4 task set and
        // still execute every expert through the ordinary CPU path.
        const auto make_mixed_mxfp4 = [](uint32_t rows, uint32_t columns, uint32_t seed) {
            TensorData tensor;
            tensor.dtype = DType::MxFp4;
            tensor.shape = {rows, columns};
            tensor.mxfp4_blocks.resize(static_cast<size_t>(rows) * columns / 2);
            tensor.mxfp4_scales.resize(static_cast<size_t>(rows) * columns / 32);
            for (size_t index = 0; index < tensor.mxfp4_blocks.size(); ++index)
                tensor.mxfp4_blocks[index] = static_cast<uint8_t>((index * 29 + seed * 7) % 256);
            std::fill_n(tensor.mxfp4_scales.data(), tensor.mxfp4_scales.size(), uint8_t{127});
            return tensor;
        };
        const auto make_mixed_bfloat16 = [](uint32_t rows, uint32_t columns, uint32_t seed) {
            TensorData tensor;
            tensor.dtype = DType::BFloat16;
            tensor.shape = {rows, columns};
            tensor.bfloat16_data.resize(tensor.element_count());
            for (size_t index = 0; index < tensor.bfloat16_data.size(); ++index)
            {
                const float value = static_cast<float>(static_cast<int>((index * 11 + seed) % 17) - 8) * 0.015625f;
                tensor.bfloat16_data[index] = float_to_bfloat16(value);
            }
            return tensor;
        };

        CompiledModel mixed_model;
        mixed_model.descriptor.hidden_size = 32;
        mixed_model.descriptor.expert_count = 2;
        mixed_model.opt.hybrid_mode = HybridMode::CpuOnly;
        mixed_model.opt.optimization_flags = g_test_optimization_flags & ~OptimizationCpuMxfp4Q8;
        MoeBlockPlan mixed_moe;
        mixed_moe.experts.resize(2);
        std::array<TensorData, 2> mixed_gate_up = {
            make_mixed_mxfp4(64, 32, 1),
            make_mixed_bfloat16(64, 32, 2),
        };
        std::array<TensorData, 2> mixed_down = {
            make_mixed_mxfp4(32, 32, 3),
            make_mixed_bfloat16(32, 32, 4),
        };
        mixed_gate_up[0].input_column_indices.resize(mixed_gate_up[0].shape[1]);
        mixed_down[0].input_column_indices.resize(mixed_down[0].shape[1]);
        for (uint32_t column = 0; column < mixed_gate_up[0].shape[1]; ++column)
        {
            mixed_gate_up[0].input_column_indices[column] = mixed_gate_up[0].shape[1] - column - 1;
            mixed_down[0].input_column_indices[column] = mixed_down[0].shape[1] - column - 1;
        }
        for (size_t expert_id = 0; expert_id < mixed_moe.experts.size(); ++expert_id)
        {
            ExpertPlan& expert = mixed_moe.experts[expert_id];
            const auto gate_up = mixed_model.weights.add("mixed.gate_up." + std::to_string(expert_id), mixed_gate_up[expert_id]);
            const auto down = mixed_model.weights.add("mixed.down." + std::to_string(expert_id), mixed_down[expert_id]);
            check(static_cast<bool>(gate_up) && static_cast<bool>(down));
            expert.layout = ExpertLayout::PackedGateUpDown;
            expert.activation = ExpertActivation::Silu;
            expert.gate_up_weight = gate_up.value();
            expert.down_weight = down.value();
            expert.gate_up_bias = invalid_tensor_handle;
            expert.down_bias = invalid_tensor_handle;
        }

        LayerState mixed_state;
        mixed_state.resize_experts(2);
        ExpertScratch mixed_scratch;
        SessionStatistics mixed_statistics;
        for (size_t pass = 0; pass < 3; ++pass)
        {
            const bool single_input = pass != 1;
            const size_t route_count = single_input ? 1 : 3;
            mixed_state.normalized.reset(single_input ? 1 : 2, 32, false);
            for (size_t row = 0; row < mixed_state.normalized.rows(); ++row)
                for (uint32_t column = 0; column < mixed_state.normalized.columns(); ++column)
                    mixed_state.normalized.row(row)[column] = static_cast<float>(static_cast<int>((row * 13 + column * 5 + pass * 7) % 23) - 11) * 0.015625f;
            mixed_state.reset();
            for (size_t expert_id = 0; expert_id < mixed_moe.experts.size(); ++expert_id)
            {
                ExpertState& active = mixed_state.active_experts()[expert_id];
                active.batch.expert_id = static_cast<uint32_t>(expert_id);
                const uint32_t token_index = single_input ? 0 : static_cast<uint32_t>(expert_id);
                active.batch.routes.assign(route_count, {token_index, 0, 1.0f});
            }
            const auto mixed_status = forward_moe(mixed_model,
                                                  mixed_moe,
                                                  mixed_state,
                                                  &mixed_statistics,
                                                  mixed_scratch,
                                                  0,
                                                  ExecutionBackend::Cpu,
                                                  true);
            check(static_cast<bool>(mixed_status));
            check(mixed_scratch.decode_tasks.size() == mixed_moe.experts.size());

            for (size_t expert_id = 0; expert_id < mixed_moe.experts.size(); ++expert_id)
            {
                const ExpertState& active = mixed_state.active_experts()[expert_id];
                const ActivationBuffer* task_input = mixed_scratch.decode_tasks[expert_id].input;
                check(task_input == (single_input ? &mixed_state.normalized : &active.input));
                check(active.input.rows() == (single_input ? 0 : route_count));
                ActivationBuffer input(route_count, 32);
                for (size_t row = 0; row < route_count; ++row)
                    std::copy_n(mixed_state.normalized.row(active.batch.routes[row].token_index), 32, input.row(row));
                const TensorData& gate_up = mixed_gate_up[expert_id];
                const TensorData& down = mixed_down[expert_id];
                ActivationBuffer activated;
                if (gate_up.dtype == DType::MxFp4 && gate_up.input_column_indices.empty())
                {
                    forward_gate_up_mxfp4(gate_up,
                                          nullptr,
                                          input,
                                          ExpertActivation::Silu,
                                          0.0f,
                                          activated,
                                          mixed_model.opt.optimization_flags);
                }
                else
                {
                    const ActivationBuffer projected = forward_linear(gate_up,
                                                                      input,
                                                                      mixed_model.opt.optimization_flags);
                    activated.reset(route_count, 32, false);
                    for (size_t row = 0; row < route_count; ++row)
                        for (uint32_t column = 0; column < 32; ++column)
                            activated.row(row)[column] = scaled_silu(projected.row(row)[column], 1.0f, mixed_model.opt.optimization_flags)
                                                         * projected.row(row)[32 + column];
                }
                const ActivationBuffer expected = forward_linear(down,
                                                                 activated,
                                                                 mixed_model.opt.optimization_flags);
                check(active.output.rows() == expected.rows());
                check(active.output.columns() == expected.columns());
                for (size_t index = 0; index < expected.values().size(); ++index)
                    check_near(active.output.values()[index], expected.values()[index], 1e-5f);
            }
        }
    }

    auto vulkan_expert = Mxfp4Expert_vulkan::create(expert_gate_up,
                                                    &expert_gate_up_bias,
                                                    expert_down,
                                                    &expert_down_bias,
                                                    expert_activation_limit,
                                                    automatic_vulkan_device_index,
                                                    ExpertActivation::GptOssSwiGlu,
                                                    vulkan_runtime,
                                                    g_test_optimization_flags);
    if (get_gpu_count() > 0)
    {
        check(static_cast<bool>(vulkan_expert));
        for (size_t token_count : {size_t(1), size_t(2), size_t(4)})
        {
            ActivationBuffer expert_input(token_count, 32);
            for (size_t row = 0; row < expert_input.rows(); ++row)
            {
                for (uint32_t column = 0; column < expert_input.columns(); ++column)
                {
                    expert_input.row(row)[column] = static_cast<float>(static_cast<int>((column * 5 + row * 7) % 17) - 8) * 0.015625f;
                }
            }
            ActivationBuffer cpu_activated;
            forward_gate_up_mxfp4(expert_gate_up,
                                  &expert_gate_up_bias,
                                  expert_input,
                                  ExpertActivation::GptOssSwiGlu,
                                  expert_activation_limit,
                                  cpu_activated,
                                  g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
            const ActivationBuffer cpu_expert = forward_linear(expert_down,
                                                               expert_down_bias,
                                                               cpu_activated,
                                                               g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
            ActivationBuffer vulkan_expert_output;
            check(static_cast<bool>(vulkan_expert->forward(expert_input, vulkan_expert_output)));
            check(static_cast<bool>(vulkan_expert_output.rows() == cpu_expert.rows()));
            check(static_cast<bool>(vulkan_expert_output.columns() == cpu_expert.columns()));
            for (size_t row = 0; row < cpu_expert.rows(); ++row)
            {
                for (uint32_t column = 0; column < cpu_expert.columns(); ++column)
                {
                    check_near(vulkan_expert_output.row(row)[column], cpu_expert.row(row)[column], 1e-3f);
                }
            }
        }
        auto silu_vulkan_expert = Mxfp4Expert_vulkan::create(expert_gate_up,
                                                             &expert_gate_up_bias,
                                                             expert_down,
                                                             &expert_down_bias,
                                                             expert_activation_limit,
                                                             automatic_vulkan_device_index,
                                                             ExpertActivation::Silu,
                                                             vulkan_runtime,
                                                             g_test_optimization_flags);
        check(static_cast<bool>(silu_vulkan_expert));
        ActivationBuffer silu_input(1, 32);
        for (uint32_t column = 0; column < silu_input.columns(); ++column)
            silu_input.row(0)[column] = static_cast<float>(static_cast<int>((column * 11) % 23) - 11) * 0.015625f;
        ActivationBuffer silu_activated;
        forward_gate_up_mxfp4(expert_gate_up,
                              &expert_gate_up_bias,
                              silu_input,
                              ExpertActivation::Silu,
                              expert_activation_limit,
                              silu_activated,
                              g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const ActivationBuffer silu_expected = forward_linear(expert_down,
                                                              expert_down_bias,
                                                              silu_activated,
                                                              g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        ActivationBuffer silu_actual;
        check(static_cast<bool>(silu_vulkan_expert->forward(silu_input, silu_actual)));
        for (uint32_t column = 0; column < silu_expected.columns(); ++column)
            check_near(silu_actual.row(0)[column], silu_expected.row(0)[column], 1e-3f);

        auto deepseek_vulkan_expert = Mxfp4Expert_vulkan::create(expert_gate_up,
                                                                 &expert_gate_up_bias,
                                                                 expert_down,
                                                                 &expert_down_bias,
                                                                 expert_activation_limit,
                                                                 automatic_vulkan_device_index,
                                                                 ExpertActivation::DeepSeekSwiGlu,
                                                                 vulkan_runtime,
                                                                 g_test_optimization_flags);
        check(static_cast<bool>(deepseek_vulkan_expert));
        ActivationBuffer deepseek_input(1, 32);
        for (uint32_t column = 0; column < deepseek_input.columns(); ++column)
            deepseek_input.row(0)[column] = static_cast<float>(static_cast<int>((column * 3) % 19) - 9) * 0.015625f;
        ActivationBuffer deepseek_activated;
        forward_gate_up_mxfp4(expert_gate_up,
                              &expert_gate_up_bias,
                              deepseek_input,
                              ExpertActivation::DeepSeekSwiGlu,
                              expert_activation_limit,
                              deepseek_activated,
                              g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const ActivationBuffer deepseek_expected = forward_linear(expert_down,
                                                                  expert_down_bias,
                                                                  deepseek_activated,
                                                                  g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        ActivationBuffer deepseek_actual;
        check(static_cast<bool>(deepseek_vulkan_expert->forward(deepseek_input, deepseek_actual)));
        for (uint32_t column = 0; column < deepseek_expected.columns(); ++column)
            check_near(deepseek_actual.row(0)[column], deepseek_expected.row(0)[column], 1e-3f);

        TensorData expert_gate_up_bias_bfloat16;
        expert_gate_up_bias_bfloat16.dtype = DType::BFloat16;
        expert_gate_up_bias_bfloat16.shape = expert_gate_up_bias.shape;
        expert_gate_up_bias_bfloat16.bfloat16_data.reserve(expert_gate_up_bias.float32_data.size());
        for (float value : expert_gate_up_bias.float32_data)
            expert_gate_up_bias_bfloat16.bfloat16_data.push_back(float_to_bfloat16(value));
        TensorData expert_down_bias_bfloat16;
        expert_down_bias_bfloat16.dtype = DType::BFloat16;
        expert_down_bias_bfloat16.shape = expert_down_bias.shape;
        expert_down_bias_bfloat16.bfloat16_data.reserve(expert_down_bias.float32_data.size());
        for (float value : expert_down_bias.float32_data)
            expert_down_bias_bfloat16.bfloat16_data.push_back(float_to_bfloat16(value));

        auto bfloat16_vulkan_expert = Mxfp4Expert_vulkan::create(expert_gate_up,
                                                                 &expert_gate_up_bias_bfloat16,
                                                                 expert_down,
                                                                 &expert_down_bias_bfloat16,
                                                                 expert_activation_limit,
                                                                 automatic_vulkan_device_index,
                                                                 ExpertActivation::DeepSeekSwiGlu,
                                                                 vulkan_runtime,
                                                                 g_test_optimization_flags);
        check(static_cast<bool>(bfloat16_vulkan_expert));
        ActivationBuffer bfloat16_activated;
        forward_gate_up_mxfp4(expert_gate_up,
                              &expert_gate_up_bias_bfloat16,
                              deepseek_input,
                              ExpertActivation::DeepSeekSwiGlu,
                              expert_activation_limit,
                              bfloat16_activated,
                              g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const ActivationBuffer bfloat16_expected = forward_linear(expert_down,
                                                                  expert_down_bias_bfloat16,
                                                                  bfloat16_activated,
                                                                  g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        ActivationBuffer bfloat16_actual;
        check(static_cast<bool>(bfloat16_vulkan_expert->forward(deepseek_input, bfloat16_actual)));
        check(static_cast<bool>(bfloat16_actual.rows() == bfloat16_expected.rows()));
        check(static_cast<bool>(bfloat16_actual.columns() == bfloat16_expected.columns()));
        for (uint32_t column = 0; column < bfloat16_expected.columns(); ++column)
            check_near(bfloat16_actual.row(0)[column], bfloat16_expected.row(0)[column], 1e-3f);
    }
    else
    {
        check(static_cast<bool>(!vulkan_expert));
    }

    const uint64_t backend_optimization_flags = g_test_optimization_flags
                                                | OptimizationVulkanRouteAggregation;
    auto expert_backend = create_vulkan_expert_backend(4096,
                                                       automatic_vulkan_device_index,
                                                       nullptr,
                                                       vulkan_runtime,
                                                       backend_optimization_flags);
    if (get_gpu_count() > 0)
    {
        check(static_cast<bool>(expert_backend));
        for (size_t aggregation_case = 0; aggregation_case < 2; ++aggregation_case)
        {
            const bool aggregate_routes = aggregation_case != 0;
            // Cold and warm Hybrid routes stay on GPU even when the former
            // priority optimization bit is disabled. Warm routes can aggregate.
            CompiledModel model;
            model.descriptor.hidden_size = expert_down.shape[0];
            model.opt.hybrid_mode = HybridMode::HybridExperts;
            model.opt.optimization_flags = g_test_optimization_flags
                                           & ~OptimizationCpuMxfp4Q8
                                           & ~OptimizationVulkanExpertGpuPriority;
            if (aggregate_routes)
                model.opt.optimization_flags |= OptimizationVulkanRouteAggregation;
            else
                model.opt.optimization_flags &= ~OptimizationVulkanRouteAggregation;
            uint64_t block_optimization_flags = backend_optimization_flags;
            if (!aggregate_routes)
                block_optimization_flags &= ~OptimizationVulkanRouteAggregation;
            auto block_backend = create_vulkan_expert_backend(4096,
                                                              automatic_vulkan_device_index,
                                                              nullptr,
                                                              vulkan_runtime,
                                                              block_optimization_flags);
            check(static_cast<bool>(block_backend));
            model.expert_backend = block_backend;
            auto gate_up = model.weights.add("block_gate_up", expert_gate_up);
            auto down = model.weights.add("block_down", expert_down);
            auto gate_bias = model.weights.add("block_gate_bias", expert_gate_up_bias);
            auto down_bias = model.weights.add("block_down_bias", expert_down_bias);
            check(gate_up && down && gate_bias && down_bias);
            model.operators.bind_weight_count(model.weights.size());
            check(model.expert_cache == nullptr);
            MoeBlockPlan moe;
            moe.experts.resize(2);
            for (size_t index = 0; index < moe.experts.size(); ++index)
            {
                ExpertPlan& expert = moe.experts[index];
                expert.gate_up_weight = gate_up.value();
                expert.down_weight = down.value();
                expert.gate_up_bias = gate_bias.value();
                expert.down_bias = down_bias.value();
                expert.activation = ExpertActivation::GptOssSwiGlu;
                expert.activation_limit = expert_activation_limit;
                expert.cache_key = "nonresident-block-" + std::to_string(index);
            }
            LayerState state;
            state.resize_experts(2);
            for (size_t index = 0; index < state.active_experts().size(); ++index)
            {
                ActivationBuffer& output = state.active_experts()[index].output;
                output.reset(65, expert_down.shape[0], false);
            }
            ExpertScratch scratch;
            SessionStatistics statistics;
            const auto block_gate_owner = std::make_shared<const TensorData>(expert_gate_up);
            const auto block_down_owner = std::make_shared<const TensorData>(expert_down);
            for (size_t pass = 0; pass < 3; ++pass)
            {
                state.normalized.reset(aggregate_routes ? (pass == 1 ? 64 : 32) : 96,
                                       expert_gate_up.shape[1],
                                       false);
                for (size_t row = 0; row < state.normalized.rows(); ++row)
                    for (uint32_t column = 0; column < state.normalized.columns(); ++column)
                        state.normalized.row(row)[column] = static_cast<float>(static_cast<int>((row + column + pass) % 17) - 8) * 0.015625f;
                for (size_t index = 0; index < 2; ++index)
                {
                    ExpertState& active = state.active_experts()[index];
                    active.batch.expert_id = static_cast<uint32_t>(index);
                    const size_t count = index == 0
                                             ? (pass == 1 ? 65 : 32)
                                             : (pass == 1 ? 16 : 8);
                    active.batch.routes.resize(count);
                    for (size_t row = 0; row < count; ++row)
                    {
                        const size_t token_count = state.normalized.rows();
                        size_t token_index = row;
                        if (index == 0 && aggregate_routes)
                        {
                            token_index = pass == 1 ? std::min(row, token_count - 1)
                                                    : row % (token_count - 1);
                        }
                        else if (index != 0)
                        {
                            token_index = row * (aggregate_routes || pass == 1 ? 3 : 5);
                        }
                        check(token_index < token_count);
                        active.batch.routes[row] = {static_cast<uint32_t>(token_index), 0, 1.0f};
                    }
                }
                ActivationBuffer expected_aggregated(state.normalized.rows(), expert_down.shape[0]);
                const uint64_t executions_before = block_backend->statistics().executions;
                const uint64_t aggregations_before = block_backend->statistics().route_aggregation_batches;
                auto ret = forward_moe(model, moe, state, &statistics, scratch, 0, ExecutionBackend::Vulkan, false);
                check(static_cast<bool>(ret));
                const bool expect_aggregated = aggregate_routes && pass != 0;
                check(block_backend->statistics().route_aggregation_batches
                      == aggregations_before + (expect_aggregated ? 1 : 0));
                check(scratch.backend_aggregated_output_valid == expect_aggregated);
                check(scratch.backend_aggregated.size() == state.active_experts().size());
                if (expect_aggregated)
                {
                    check(scratch.backend_aggregated[0] == 1);
                    check(scratch.backend_aggregated[1] == 1);
                }
                else
                {
                    check(scratch.backend_aggregated[0] == 0);
                    check(scratch.backend_aggregated[1] == 0);
                }
                const bool gpu_executed = block_backend->statistics().executions > executions_before;
                check(gpu_executed);
                check(statistics.expert_cpu_batches == 0 && statistics.expert_cpu_routes == 0
                      && statistics.expert_cpu_fallbacks == 0);
                const float tolerance = 1e-3f;
                for (size_t index = 0; index < state.active_experts().size(); ++index)
                {
                    const ExpertState& active = state.active_experts()[index];
                    ActivationBuffer input(active.batch.routes.size(), state.normalized.columns());
                    for (size_t row = 0; row < input.rows(); ++row)
                        std::copy_n(state.normalized.row(active.batch.routes[row].token_index), input.columns(), input.row(row));
                    ActivationBuffer activated;
                    forward_gate_up_mxfp4(expert_gate_up,
                                          &expert_gate_up_bias,
                                          input,
                                          ExpertActivation::GptOssSwiGlu,
                                          expert_activation_limit,
                                          activated,
                                          model.opt.optimization_flags);
                    const ActivationBuffer expected = forward_linear(expert_down,
                                                                     expert_down_bias,
                                                                     activated,
                                                                     model.opt.optimization_flags);
                    if (!expect_aggregated)
                    {
                        check(active.output.rows() == input.rows());
                        check(active.output.columns() == expected.columns());
                        for (size_t value = 0; value < expected.values().size(); ++value)
                            check_near(active.output.values()[value], expected.values()[value], tolerance);
                    }
                    for (size_t row = 0; row < active.batch.routes.size(); ++row)
                    {
                        float* aggregate_row = expected_aggregated.row(active.batch.routes[row].token_index);
                        const float* expected_row = expected.row(row);
                        for (uint32_t column = 0; column < expected.columns(); ++column)
                            aggregate_row[column] += expected_row[column] * active.batch.routes[row].weight;
                    }
                }
                if (expect_aggregated)
                {
                    check(scratch.backend_aggregated_output.rows() == expected_aggregated.rows());
                    check(scratch.backend_aggregated_output.columns() == expected_aggregated.columns());
                    for (size_t value = 0; value < expected_aggregated.values().size(); ++value)
                        check_near(scratch.backend_aggregated_output.values()[value],
                                   expected_aggregated.values()[value],
                                   1e-3f);
                    if (pass != 1)
                        for (uint32_t column = 0; column < expected_aggregated.columns(); ++column)
                            check_near(scratch.backend_aggregated_output.row(31)[column], 0.0f, 1e-6f);
                }
                state.reset();
                if (pass == 0)
                {
                    const size_t admitted_experts = aggregate_routes ? 1 : moe.experts.size();
                    for (size_t index = 0; index < admitted_experts; ++index)
                        block_backend->admit(ExpertDemandRequest{"nonresident-block-" + std::to_string(index),
                                                                 block_gate_owner,
                                                                 &expert_gate_up_bias,
                                                                 block_down_owner,
                                                                 &expert_down_bias,
                                                                 0,
                                                                 expert_activation_limit,
                                                                 ExpertActivation::GptOssSwiGlu});
                    block_backend->wait_for_background_work();
                }
                else if (aggregate_routes && pass == 1)
                {
                    block_backend->admit(ExpertDemandRequest{"nonresident-block-1", block_gate_owner,
                                                             &expert_gate_up_bias, block_down_owner,
                                                             &expert_down_bias, 0, expert_activation_limit,
                                                             ExpertActivation::GptOssSwiGlu});
                    block_backend->wait_for_background_work();
                }
            }
        }
        const auto backend_gate_up = std::make_shared<TensorData>(expert_gate_up);
        const auto backend_down = std::make_shared<TensorData>(expert_down);
        const ExpertDemandRequest expert_admission{"test-expert", backend_gate_up, &expert_gate_up_bias,
                                                   backend_down, &expert_down_bias, 0, expert_activation_limit};
        expert_backend->admit(expert_admission);
        expert_backend->admit(expert_admission);
        const auto admission_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (expert_backend->statistics().stores == 0 && std::chrono::steady_clock::now() < admission_deadline)
        {
            std::this_thread::yield();
        }
        check(static_cast<bool>(expert_backend->statistics().stores == 1));
        ActivationBuffer backend_input(1, 32);
        for (uint32_t column = 0; column < backend_input.columns(); ++column)
        {
            backend_input.row(0)[column] = static_cast<float>(static_cast<int>(column % 13) - 6) * 0.015625f;
        }
        ActivationBuffer backend_activated;
        forward_gate_up_mxfp4(expert_gate_up,
                              &expert_gate_up_bias,
                              backend_input,
                              ExpertActivation::GptOssSwiGlu,
                              expert_activation_limit,
                              backend_activated,
                              g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const ActivationBuffer backend_expected = forward_linear(expert_down,
                                                                 expert_down_bias,
                                                                 backend_activated,
                                                                 g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        for (uint32_t sample = 0; sample < 3; ++sample)
        {
            ActivationBuffer backend_output;
            const ExpertRequest request{"test-expert", &backend_input, &backend_output};
            const auto results = submit_backend_batch(*expert_backend, std::span<const ExpertRequest>(&request, 1));
            check(results.size() == 1);
            check(results.front() == ExpertStatus::Executed);
            for (uint32_t column = 0; column < backend_expected.columns(); ++column)
            {
                check_near(backend_output.row(0)[column], backend_expected.row(0)[column], 1e-3f);
            }
        }
        auto backend_gate_up_second = std::make_shared<TensorData>(expert_gate_up);
        auto backend_down_second = std::make_shared<TensorData>(expert_down);
        backend_gate_up_second->mxfp4_blocks[0] ^= 1;
        backend_down_second->mxfp4_blocks[0] ^= 1;
        const ExpertDemandRequest second_expert_admission{"test-expert-second", backend_gate_up_second,
                                                          &expert_gate_up_bias, backend_down_second,
                                                          &expert_down_bias, 0, expert_activation_limit};
        expert_backend->admit(second_expert_admission);
        expert_backend->admit(second_expert_admission);
        const auto second_admission_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (expert_backend->statistics().stores < 2 && std::chrono::steady_clock::now() < second_admission_deadline)
        {
            std::this_thread::yield();
        }
        check(static_cast<bool>(expert_backend->statistics().stores == 2));
        ActivationBuffer backend_second_activated;
        forward_gate_up_mxfp4(*backend_gate_up_second,
                              &expert_gate_up_bias,
                              backend_input,
                              ExpertActivation::GptOssSwiGlu,
                              expert_activation_limit,
                              backend_second_activated,
                              g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const ActivationBuffer backend_second_expected = forward_linear(*backend_down_second,
                                                                        expert_down_bias,
                                                                        backend_second_activated,
                                                                        g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        ActivationBuffer backend_batch_output_first;
        ActivationBuffer backend_batch_output_second;
        const std::array<ExpertRequest, 2> backend_requests = {{
            {
                "test-expert",
                &backend_input,
                &backend_batch_output_first,
            },
            {
                "test-expert-second",
                &backend_input,
                &backend_batch_output_second,
            },
        }};
        const auto backend_batch_results = submit_backend_batch(*expert_backend, backend_requests);
        check(static_cast<bool>(backend_batch_results.size() == 2));
        for (ExpertStatus result : backend_batch_results)
        {
            check(static_cast<bool>(result == ExpertStatus::Executed));
        }
        for (uint32_t column = 0; column < backend_expected.columns(); ++column)
        {
            check_near(backend_batch_output_first.row(0)[column], backend_expected.row(0)[column], 1e-3f);
            check_near(backend_batch_output_second.row(0)[column], backend_second_expected.row(0)[column], 1e-3f);
        }

        const ExpertBackendStatistics backend_statistics = expert_backend->statistics();
        check(static_cast<bool>(backend_statistics.executions == 5));
        check(static_cast<bool>(backend_statistics.hits == 5));
        check(static_cast<bool>(backend_statistics.bytes_uploaded > 0));
        check(static_cast<bool>(backend_statistics.arc_frequent_size > 0));

        {
            // A scan in one layer must evict an unused Recent entry in
            // another layer before evicting that layer's reused Experts.
            const uint64_t pair_size = quant_expert_bytes(*backend_gate_up, nullptr, *backend_down,
                                                          automatic_vulkan_device_index,
                                                          vulkan_runtime, backend_optimization_flags,
                                                          &expert_gate_up_bias, nullptr, &expert_down_bias,
                                                          ExpertActivation::GptOssSwiGlu, expert_activation_limit);
            check(pair_size != 0);
            auto arc_backend = create_vulkan_expert_backend(pair_size * 3,
                                                            automatic_vulkan_device_index,
                                                            nullptr,
                                                            vulkan_runtime,
                                                            backend_optimization_flags);
            check(static_cast<bool>(arc_backend));
            const auto admit_arc_expert = [&](const char* key, uint32_t group) {
                arc_backend->admit(ExpertDemandRequest{key, backend_gate_up, &expert_gate_up_bias,
                                                       backend_down, &expert_down_bias, group,
                                                       expert_activation_limit});
                arc_backend->wait_for_background_work();
            };
            admit_arc_expert("arc-hot-0", 0);
            admit_arc_expert("arc-hot-1", 0);
            admit_arc_expert("arc-cold-0", 1);
            {
                std::array<ActivationBuffer, 2> hot_outputs;
                const std::array<ExpertRequest, 2> hot_requests = {{
                    {"arc-hot-0", &backend_input, &hot_outputs[0]},
                    {"arc-hot-1", &backend_input, &hot_outputs[1]},
                }};
                const auto results = submit_backend_batch(*arc_backend, hot_requests);
                check(results.size() == 2);
                check(std::all_of(results.begin(), results.end(), [](ExpertStatus result) {
                    return result == ExpertStatus::Executed;
                }));
            }
            const auto hot_statistics = arc_backend->statistics();
            check(hot_statistics.arc_frequent_size == pair_size * 2);
            check(hot_statistics.arc_recent_size == pair_size);
            admit_arc_expert("arc-new-0", 0);
            const auto admitted_statistics = arc_backend->statistics();
            check(admitted_statistics.stores == 4);
            check(admitted_statistics.evictions == 1);
            check(admitted_statistics.resident_size == pair_size * 3);
            check(admitted_statistics.pending_size == 0);
            check(admitted_statistics.arc_frequent_size == pair_size * 2);
            check(admitted_statistics.arc_recent_size == pair_size);
            {
                std::array<ActivationBuffer, 4> outputs;
                const std::array<ExpertRequest, 4> requests = {{
                    {"arc-hot-0", &backend_input, &outputs[0]},
                    {"arc-hot-1", &backend_input, &outputs[1]},
                    {"arc-cold-0", &backend_input, &outputs[2]},
                    {"arc-new-0", &backend_input, &outputs[3]},
                }};
                const auto results = submit_backend_batch(*arc_backend, requests);
                check(results.size() == requests.size());
                check(results[0] == ExpertStatus::Executed);
                check(results[1] == ExpertStatus::Executed);
                check(results[2] == ExpertStatus::NotResident);
                check(results[3] == ExpertStatus::Executed);
                for (size_t index : {size_t{0}, size_t{1}, size_t{3}})
                {
                    check(outputs[index].rows() == backend_expected.rows());
                    check(outputs[index].columns() == backend_expected.columns());
                    for (uint32_t column = 0; column < backend_expected.columns(); ++column)
                        check_near(outputs[index].row(0)[column], backend_expected.row(0)[column], 1e-3f);
                }
                check(outputs[2].rows() == 0);
            }
            check(arc_backend->statistics().execution_failures == 0);
        }
        for (const int scenario : {0, 1, 2})
        {
            const float sentinel = 91.0f + static_cast<float>(scenario);
            ActivationBuffer guarded_output(1, backend_expected.columns());
            std::fill_n(guarded_output.row(0), guarded_output.columns(), sentinel);
            const auto check_guarded_output = [&] {
                check(guarded_output.rows() == 1);
                check(guarded_output.columns() == backend_expected.columns());
                check(std::all_of(guarded_output.values().begin(),
                                  guarded_output.values().end(),
                                  [sentinel](float value) { return value == sentinel; }));
            };
            const ExpertRequest single_request{
                "test-expert",
                &backend_input,
                &guarded_output,
            };
            const uint64_t executions_before = expert_backend->statistics().executions;
            auto single_submission = expert_backend->submit_batch(std::span<const ExpertRequest>(&single_request, 1));
            check(static_cast<bool>(single_submission));
            const std::span<const ExpertStatus> reservations = single_submission->reservations();
            check(reservations.size() == 1);
            check(reservations.front() == ExpertStatus::Executed);
            check_guarded_output();

            if (scenario == 0)
            {
                single_submission->abort();
                check_guarded_output();
                const auto results = single_submission->wait();
                check(results.size() == 1);
                check(results.front() == ExpertStatus::Executed);
                check(expert_backend->statistics().executions == executions_before + 1);
                check_guarded_output();
                check(!single_submission->commit());
                check_guarded_output();
                single_submission->abort();
                check_guarded_output();
            }
            else if (scenario == 1)
            {
                const auto results = single_submission->wait();
                check(results.size() == 1);
                check(results.front() == ExpertStatus::Executed);
                check(expert_backend->statistics().executions == executions_before + 1);
                check_guarded_output();
                single_submission->abort();
                check(!single_submission->commit());
                check_guarded_output();
                single_submission->abort();
                check_guarded_output();
            }
            else
            {
                const auto results = single_submission->wait();
                check(results.size() == 1);
                check(results.front() == ExpertStatus::Executed);
                check(expert_backend->statistics().executions == executions_before + 1);
                check_guarded_output();
                check(single_submission->commit());
                check(guarded_output.rows() == backend_expected.rows());
                check(guarded_output.columns() == backend_expected.columns());
                for (uint32_t column = 0; column < backend_expected.columns(); ++column)
                    check_near(guarded_output.row(0)[column], backend_expected.row(0)[column], 1e-3f);
                single_submission->abort();
                for (uint32_t column = 0; column < backend_expected.columns(); ++column)
                    check_near(guarded_output.row(0)[column], backend_expected.row(0)[column], 1e-3f);
            }
        }

        auto device_source = create_vulkan_victim_cache(4096,
                                                        automatic_vulkan_device_index,
                                                        vulkan_runtime,
                                                        backend_optimization_flags);
        check(static_cast<bool>(device_source));
        device_source->admit("victim-expert", backend_gate_up, backend_down,
                             {
                                 &expert_gate_up_bias,
                                 &expert_down_bias,
                                 expert_activation_limit,
                                 true,
                             });
        device_source->wait_for_background_work();
        check(static_cast<bool>(device_source->statistics().stores == 1));
        auto source_backend = create_vulkan_expert_backend(0,
                                                           automatic_vulkan_device_index,
                                                           device_source,
                                                           vulkan_runtime,
                                                           backend_optimization_flags);
        check(static_cast<bool>(source_backend));
        ActivationBuffer source_output;
        const ExpertRequest source_request{"victim-expert", &backend_input, &source_output};
        const auto source_results = submit_backend_batch(*source_backend, std::span<const ExpertRequest>(&source_request, 1));
        check(source_results.size() == 1);
        check(source_results.front() == ExpertStatus::Executed);
        for (uint32_t column = 0; column < backend_expected.columns(); ++column)
        {
            check_near(source_output.row(0)[column], backend_expected.row(0)[column], 1e-3f);
        }
        const ExpertBackendStatistics source_statistics = source_backend->statistics();
        check(static_cast<bool>(source_statistics.device_source_hits == 1));
        check(static_cast<bool>(source_statistics.device_source_executions == 1));
        check(static_cast<bool>(source_statistics.device_source_execution_failures == 0));
        check(static_cast<bool>(device_source->statistics().bytes_downloaded == 0));
    }
    else
    {
        check(static_cast<bool>(!expert_backend));
    }

    constexpr uint32_t test_block_count = 4;
    constexpr size_t test_token_count = 3;
    constexpr size_t test_input_columns = test_block_count * 32;
    constexpr size_t test_input_stride = test_input_columns + 5;
    constexpr size_t test_output_stride = 3;
    std::vector<uint8_t> first_packed(test_block_count * 16);
    std::vector<uint8_t> second_packed(test_block_count * 16);
    for (size_t index = 0; index < first_packed.size(); ++index)
    {
        first_packed[index] = static_cast<uint8_t>(((index * 5 + 3) % 16) | (((index * 7 + 1) % 16) << 4));
        second_packed[index] = static_cast<uint8_t>(((index * 11 + 2) % 16) | (((index * 3 + 9) % 16) << 4));
    }
    const std::array<uint8_t, test_block_count> first_scales = {125, 127, 128, 126};
    const std::array<uint8_t, test_block_count> second_scales = {128, 124, 127, 129};
    std::vector<float> strided_input(test_token_count * test_input_stride);
    for (size_t token = 0; token < test_token_count; ++token)
    {
        for (size_t column = 0; column < test_input_stride; ++column)
        {
            strided_input[token * test_input_stride + column] = static_cast<float>(static_cast<int>((column * 13 + token * 5) % 23) - 11) * 0.03125f;
        }
    }
    auto reference_dot = [&](const std::vector<uint8_t>& packed, const std::array<uint8_t, test_block_count>& scales, size_t token) {
        float sum = 0.0f;
        for (uint32_t block = 0; block < test_block_count; ++block)
        {
            float block_sum = 0.0f;
            for (uint32_t byte = 0; byte < 16; ++byte)
            {
                const uint8_t value = packed[block * 16 + byte];
                const size_t input_offset = token * test_input_stride + block * 32 + byte * 2;
                block_sum += values[value & 0x0f] * strided_input[input_offset];
                block_sum += values[value >> 4] * strided_input[input_offset + 1];
            }
            sum += block_sum * std::ldexp(1.0f, static_cast<int>(scales[block]) - 127);
        }
        return sum;
    };

    check_near(mxfp4_dot(first_packed.data(), first_scales.data(), test_block_count, strided_input.data()), reference_dot(first_packed, first_scales, 0), 1e-4f);

    std::vector<float> gemm_output(test_token_count * test_output_stride, -999.0f);
    mxfp4_gemm_row(first_packed.data(), first_scales.data(), test_block_count, strided_input.data(), test_input_stride, test_token_count, gemm_output.data(), test_output_stride);
    for (size_t token = 0; token < test_token_count; ++token)
    {
        check_near(gemm_output[token * test_output_stride], reference_dot(first_packed, first_scales, token), 1e-4f);
    }

    std::vector<float> paired_first(test_token_count * test_output_stride, -999.0f);
    std::vector<float> paired_second(test_token_count * test_output_stride, -999.0f);
    mxfp4_matmul_rows2(first_packed.data(), first_scales.data(), second_packed.data(), second_scales.data(), test_block_count, strided_input.data(), test_input_stride, test_token_count, paired_first.data(), test_output_stride,
                       paired_second.data(), test_output_stride);
    for (size_t token = 0; token < test_token_count; ++token)
    {
        check_near(paired_first[token * test_output_stride], reference_dot(first_packed, first_scales, token), 1e-4f);
        check_near(paired_second[token * test_output_stride], reference_dot(second_packed, second_scales, token), 1e-4f);
    }

    constexpr uint32_t bulk_pair_count = 2;
    constexpr size_t bulk_output_stride = 7;
    std::vector<uint8_t> bulk_packed(bulk_pair_count * 2 * test_block_count * 16);
    std::vector<uint8_t> bulk_scales(bulk_pair_count * 2 * test_block_count);
    const size_t packed_row_bytes = test_block_count * 16;
    for (uint32_t pair = 0; pair < bulk_pair_count; ++pair)
    {
        const std::vector<uint8_t>& first_source = pair == 0 ? first_packed : second_packed;
        const std::vector<uint8_t>& second_source = pair == 0 ? second_packed : first_packed;
        const auto& first_scale_source = pair == 0 ? first_scales : second_scales;
        const auto& second_scale_source = pair == 0 ? second_scales : first_scales;
        std::copy(first_source.begin(), first_source.end(), bulk_packed.begin() + pair * 2 * packed_row_bytes);
        std::copy(second_source.begin(), second_source.end(), bulk_packed.begin() + (pair * 2 + 1) * packed_row_bytes);
        std::copy(first_scale_source.begin(), first_scale_source.end(), bulk_scales.begin() + pair * 2 * test_block_count);
        std::copy(second_scale_source.begin(), second_scale_source.end(), bulk_scales.begin() + (pair * 2 + 1) * test_block_count);
    }
    std::vector<float> bulk_output(test_token_count * bulk_output_stride, -999.0f);
    mxfp4_matmul_row_pairs(bulk_packed.data(), bulk_scales.data(), test_block_count, bulk_pair_count, strided_input.data(), test_input_stride, test_token_count, bulk_output.data() + 1, 2, bulk_output_stride, bulk_output.data() + 2, 2,
                           bulk_output_stride);
    for (size_t token = 0; token < test_token_count; ++token)
    {
        check_near(bulk_output[token * bulk_output_stride + 1], reference_dot(first_packed, first_scales, token), 1e-4f);
        check_near(bulk_output[token * bulk_output_stride + 2], reference_dot(second_packed, second_scales, token), 1e-4f);
        check_near(bulk_output[token * bulk_output_stride + 3], reference_dot(second_packed, second_scales, token), 1e-4f);
        check_near(bulk_output[token * bulk_output_stride + 4], reference_dot(first_packed, first_scales, token), 1e-4f);
    }

    constexpr size_t paired_token_count = 2;
    std::vector<float> paired_bulk_output(paired_token_count * bulk_output_stride, -999.0f);
    mxfp4_matmul_row_pairs(bulk_packed.data(), bulk_scales.data(), test_block_count, bulk_pair_count, strided_input.data(), test_input_stride, paired_token_count, paired_bulk_output.data() + 1, 2, bulk_output_stride,
                           paired_bulk_output.data() + 2, 2, bulk_output_stride);
    for (size_t token = 0; token < paired_token_count; ++token)
    {
        check_near(paired_bulk_output[token * bulk_output_stride + 1], reference_dot(first_packed, first_scales, token), 1e-4f);
        check_near(paired_bulk_output[token * bulk_output_stride + 2], reference_dot(second_packed, second_scales, token), 1e-4f);
        check_near(paired_bulk_output[token * bulk_output_stride + 3], reference_dot(second_packed, second_scales, token), 1e-4f);
        check_near(paired_bulk_output[token * bulk_output_stride + 4], reference_dot(first_packed, first_scales, token), 1e-4f);
    }

    constexpr size_t packed_matrix_rows = 13;
    std::vector<uint8_t> matrix_packed(packed_matrix_rows * test_block_count * 16);
    std::vector<uint8_t> matrix_scales(packed_matrix_rows * test_block_count);
    for (size_t row = 0; row < packed_matrix_rows; ++row)
    {
        for (uint32_t block = 0; block < test_block_count; ++block)
        {
            matrix_scales[row * test_block_count + block] = static_cast<uint8_t>(123 + ((row * 3 + block * 5) % 9));
            for (uint32_t byte = 0; byte < 16; ++byte)
            {
                const size_t offset = (row * test_block_count + block) * 16 + byte;
                matrix_packed[offset] = static_cast<uint8_t>(((row * 7 + block * 11 + byte * 3) % 16)
                                                             | (((row * 13 + block * 5 + byte * 9) % 16) << 4));
            }
        }
    }
    Mxfp4Q8PackedMatrix packed_matrix;
    check(static_cast<bool>(mxfp4_q8_pack_weights(matrix_packed.data(),
                                                  matrix_scales.data(),
                                                  test_block_count,
                                                  packed_matrix_rows,
                                                  packed_matrix)));
    check(static_cast<bool>(packed_matrix.valid()));
    check(static_cast<bool>(packed_matrix.storage.size()
                            == mxfp4_q8_packed_storage_bytes(packed_matrix_rows,
                                                             test_block_count)));

    std::array<float, 37> q8_reference_input = {};
    for (size_t index = 0; index < q8_reference_input.size(); ++index)
        q8_reference_input[index] = (static_cast<float>((index * 13) % 23) - 11.0f) * 0.37f;
    q8_reference_input[7] = 9.75f;
    q8_reference_input[34] = -4.5f;
    std::array<int8_t, 37> q8_reference_values = {};
    std::array<float, 2> q8_reference_scales = {};
    mxfp4_q8_quantize(q8_reference_input.data(),
                      q8_reference_values.data(),
                      q8_reference_scales.data(),
                      static_cast<uint32_t>(q8_reference_input.size()));
    for (uint32_t block = 0; block < q8_reference_scales.size(); ++block)
    {
        const uint32_t begin = block * 32;
        const uint32_t end = std::min<uint32_t>(static_cast<uint32_t>(q8_reference_input.size()),
                                                begin + 32);
        float maximum = 0.0f;
        for (uint32_t index = begin; index < end; ++index)
            maximum = std::max(maximum, std::fabs(q8_reference_input[index]));
        const float expected_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        check_near(q8_reference_scales[block], expected_scale, 1e-7f);
        for (uint32_t index = begin; index < end; ++index)
        {
            const float normalized = std::clamp(q8_reference_input[index] / expected_scale,
                                                -127.0f,
                                                127.0f);
            check(q8_reference_values[index] == static_cast<int8_t>(std::lrintf(normalized)));
        }
    }

    Mxfp4Q8Batch quantized_input;
    mxfp4_q8_quantize_batch(strided_input.data(),
                            test_input_stride,
                            test_token_count,
                            test_input_columns,
                            quantized_input);
    std::vector<float> packed_gemm_output(test_token_count * (packed_matrix_rows + 2), -999.0f);
    mxfp4_q8_packed_gemm(packed_matrix,
                         quantized_input.row(0),
                         quantized_input.columns,
                         quantized_input.row_scales(0),
                         (quantized_input.columns + 31) / 32,
                         test_token_count,
                         packed_gemm_output.data(),
                         packed_matrix_rows + 2);
    for (size_t token = 0; token < test_token_count; ++token)
    {
        for (size_t row = 0; row < packed_matrix_rows; ++row)
        {
            check_near(packed_gemm_output[token * (packed_matrix_rows + 2) + row],
                       mxfp4_q8_dot(matrix_packed.data() + row * test_block_count * 16,
                                    matrix_scales.data() + row * test_block_count,
                                    test_block_count,
                                    quantized_input.row(token),
                                    quantized_input.row_scales(token)),
                       1e-4f);
        }
    }
    std::vector<float> packed_gemv_output(packed_matrix_rows, -999.0f);
    mxfp4_q8_packed_gemv(packed_matrix,
                         quantized_input.row(1),
                         quantized_input.row_scales(1),
                         packed_gemv_output.data());
    for (size_t row = 0; row < packed_matrix_rows; ++row)
    {
        check_near(packed_gemv_output[row],
                   mxfp4_q8_dot(matrix_packed.data() + row * test_block_count * 16,
                                matrix_scales.data() + row * test_block_count,
                                test_block_count,
                                quantized_input.row(1),
                                quantized_input.row_scales(1)),
                   1e-4f);
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    check(static_cast<bool>(mxfp4_kernel_kind() == MxFp4KernelKind::ArmNeon || mxfp4_kernel_kind() == MxFp4KernelKind::ArmSve2));
#endif
    check(static_cast<bool>(std::string(mxfp4_kernel_name()).size() > 0));
    const std::string activation_kernel_name = scaled_silu_kernel_name(g_test_optimization_flags);
    check(!activation_kernel_name.empty());

    {
        TensorData mapped_gate_up = expert_gate_up;
        mapped_gate_up.input_column_indices.resize(mapped_gate_up.shape[1]);
        for (uint32_t column = 0; column < mapped_gate_up.shape[1]; ++column)
            mapped_gate_up.input_column_indices[column] = mapped_gate_up.shape[1] - column - 1;

        bool fused_rejected_map = false;
        ActivationBuffer rejected_output;
        try
        {
            forward_gate_up_mxfp4(mapped_gate_up, nullptr, repeated_expert_input,
                                  ExpertActivation::Silu, 0.0f, rejected_output, 0);
        }
        catch (const std::runtime_error&)
        {
            fused_rejected_map = true;
        }
        check(fused_rejected_map);

        ActivationBuffer grouped_output(1, expert_down.shape[0]);
        std::fill_n(grouped_output.row(0), grouped_output.columns(), 17.0f);
        Mxfp4Task grouped_task;
        grouped_task.gate_up = &mapped_gate_up;
        grouped_task.down = &expert_down;
        grouped_task.input = &repeated_expert_input;
        grouped_task.output = &grouped_output;
        grouped_task.activation = ExpertActivation::Silu;
        check(!forward_experts_mxfp4(std::span<const Mxfp4Task>(&grouped_task, 1), nullptr, 0));
        check(std::all_of(grouped_output.values().begin(), grouped_output.values().end(),
                          [](float value) { return value == 17.0f; }));
    }
}

void test_qnk_cpu_kernel()
{
    // Q8_K keeps the first signed maximum and honors the caller's rounding mode.
    // Unaligned blocks also verify the encoded per-16-value sums.
    const int rounding = std::fegetround();
    std::array<float, qnk_block_elements * 2 + 1> q8_input{};
    std::array<uint8_t, 292 * 2 + 1> q8_encoded{};
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    {
        check(std::fesetround(mode) == 0);
        for (int sign : {-1, 1})
        {
            for (size_t index = 1; index < q8_input.size(); ++index)
                q8_input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.25f;
            for (size_t block = 0; block < 2; ++block)
            {
                q8_input[1 + block * qnk_block_elements + 7] = 127.0f * sign;
                q8_input[1 + block * qnk_block_elements + 13] = -127.0f * sign;
            }
            qnk_q8k_quantize(q8_input.data() + 1, q8_encoded.data() + 1, qnk_block_elements * 2);
            for (size_t block = 0; block < 2; ++block)
            {
                const uint8_t* encoded = q8_encoded.data() + 1 + block * 292;
                float scale = 0.0f;
                std::memcpy(&scale, encoded, sizeof(scale));
                check(scale == static_cast<float>(-sign));
                for (size_t group = 0; group < 16; ++group)
                {
                    int16_t sum = 0;
                    for (size_t lane = 0; lane < 16; ++lane)
                    {
                        const size_t index = group * 16 + lane;
                        const int8_t value = static_cast<int8_t>(std::lrintf(-sign * q8_input[1 + block * qnk_block_elements + index]));
                        check(static_cast<int8_t>(encoded[4 + index]) == value);
                        sum = static_cast<int16_t>(sum + value);
                    }
                    int16_t actual = 0;
                    std::memcpy(&actual, encoded + 260 + group * sizeof(actual), sizeof(actual));
                    check(actual == sum);
                }
            }
        }
    }
    check(std::fesetround(rounding) == 0);
    q8_input.fill(-0.0f);
    q8_encoded.fill(0xff);
    qnk_q8k_quantize(q8_input.data() + 1, q8_encoded.data() + 1, qnk_block_elements * 2);
    check(q8_encoded[0] == 0xff);
    check(std::all_of(q8_encoded.begin() + 1, q8_encoded.end(), [](uint8_t value) { return value == 0; }));

    // Each Q6_K scale covers 16 values, including the second half of each
    // 32-value lane group. Keep this oracle independent of the kernel decoder.
    std::array<uint8_t, 210> q6{};
    std::fill_n(q6.data(), 128, uint8_t{0x11});
    std::fill_n(q6.data() + 128, 64, uint8_t{0xaa});
    for (uint32_t group = 0; group < 16; ++group)
        q6[192 + group] = static_cast<uint8_t>(static_cast<int>(group + 1) * (group & 1u ? -1 : 1));
    const std::array<uint16_t, 4> half_scales = {0x3c00, 0x0001, 0x8001, 0x0000};
    const std::array<float, 4> scales = {1.0f, 0x1p-24f, -0x1p-24f, 0.0f};
    for (size_t index = 0; index < half_scales.size(); ++index)
    {
        std::memcpy(q6.data() + 208, &half_scales[index], sizeof(uint16_t));
        std::array<float, qnk_block_elements> decoded{};
        std::array<float, qnk_block_elements> values{};
        qnk_dequantize_block(DType::Q6K, q6.data(), decoded.data());
        float expected = 0.0f;
        for (uint32_t column = 0; column < qnk_block_elements; ++column)
        {
            const uint32_t group = column / 16;
            const float weight = scales[index] * static_cast<float>(static_cast<int>(group + 1) * (group & 1u ? -1 : 1));
            check(decoded[column] == weight);
            values[column] = static_cast<float>(static_cast<int>(column % 13) - 6) * 0.03125f;
            expected += weight * values[column];
        }
        check_near(qnk_dot_block(DType::Q6K, q6.data(), values.data()), expected,
                   2e-6f * std::max(std::fabs(expected), 0x1p-24f));
    }

    constexpr uint32_t columns = 512;
    constexpr size_t rows = 9;
    constexpr size_t input_rows = 3;
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };

    ActivationBuffer input(input_rows, columns);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>(column % 17) - 8) * 0.01f * static_cast<float>(row + 1);
    }

    std::vector<uint8_t> q8_row(static_cast<size_t>(qnk_storage_bytes(DType::Q8K, 1, columns)));
    qnk_q8k_quantize(input.row(0), q8_row.data(), columns);
    alignas(64) float q8_decoded[qnk_block_elements];
    qnk_dequantize_block(DType::Q8K, q8_row.data(), q8_decoded);
    float q8_scale = 0.0f;
    std::memcpy(&q8_scale, q8_row.data(), sizeof(q8_scale));
    check(std::isfinite(q8_scale));
    check(std::fabs(q8_scale) > 0.0f);
    for (uint32_t column = 0; column < qnk_block_elements; ++column)
        check(std::fabs(q8_decoded[column] - input.row(0)[column]) <= std::fabs(q8_scale) * 0.6f);
    std::vector<uint8_t> q8_batch(input.rows() * q8_row.size());
    for (size_t row = 0; row < input.rows(); ++row)
        qnk_q8k_quantize(input.row(row), q8_batch.data() + row * q8_row.size(), columns);

    for (const DType dtype : dtypes)
    {
        const size_t block_bytes = qnk_block_bytes(dtype);
        const uint32_t block_count = columns / qnk_block_elements;
        std::vector<uint8_t> raw(static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns)));
        for (size_t index = 0; index < raw.size(); ++index)
            raw[index] = static_cast<uint8_t>((index * 37u + static_cast<uint32_t>(dtype) * 11u + 5u) & 0xffu);

        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + (row * block_count + block) * block_bytes;
                if (dtype == DType::Q2K)
                {
                    std::fill_n(encoded, 16, uint8_t{1});
                    encoded[80] = 0x00;
                    encoded[81] = 0x3c;
                    encoded[82] = 0;
                    encoded[83] = 0;
                }
                else if (dtype == DType::Q3K)
                {
                    std::fill_n(encoded + 96, 12, uint8_t{0});
                    encoded[108] = 0x00;
                    encoded[109] = 0x3c;
                }
                else if (dtype == DType::Q4K || dtype == DType::Q5K)
                {
                    encoded[0] = 0x00;
                    encoded[1] = 0x3c;
                    encoded[2] = 0;
                    encoded[3] = 0;
                    std::fill_n(encoded + 4, 4, uint8_t{1});
                    std::fill_n(encoded + 8, 4, uint8_t{0});
                }
                else if (dtype == DType::Q6K)
                {
                    std::fill_n(encoded + 192, 16, uint8_t{1});
                    encoded[208] = 0x00;
                    encoded[209] = 0x3c;
                }
                else
                {
                    const float scale = 1.0f;
                    std::memcpy(encoded, &scale, sizeof(scale));
                }
            }
        }

        QnKPack packed;
        check(qnk_pack_weights(raw.data(), raw.size(), dtype, rows, columns, packed));
        check(packed.valid());
        check(packed.block_count == block_count);
        check(packed.storage.size() == ((rows + 7) / 8) * block_count * 8 * block_bytes);
        check(packed.storage.size() == qnk_packed_storage_bytes(dtype, rows, columns));
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                const uint8_t* original = raw.data() + (row * block_count + block) * block_bytes;
                check(std::memcmp(qnk_packed_block(packed, row, block), original, block_bytes) == 0);
            }
        }

        TensorData matrix;
        matrix.dtype = dtype;
        matrix.shape = {static_cast<uint32_t>(rows), columns};
        matrix.quantized_data = raw;
        ActivationBuffer projected;
        forward_linear(matrix, input, projected, 0);
        check(projected.rows() == input_rows);
        check(projected.columns() == rows);
        ActivationBuffer packed_projected;
        forward_linear(matrix,
                       input,
                       packed_projected,
                       OptimizationCpuPackedWeights);
        CompiledOperator packed_owner;
        ActivationBuffer owner_unpacked_projected;
        forward_linear(matrix,
                       input,
                       owner_unpacked_projected,
                       0,
                       &packed_owner,
                       ExecutionBackend::Cpu);
        check(static_cast<bool>(!packed_owner.qnk_packed));
        ActivationBuffer owner_packed_projected;
        forward_linear(matrix,
                       input,
                       owner_packed_projected,
                       OptimizationCpuPackedWeights,
                       &packed_owner,
                       ExecutionBackend::Cpu);
        check(static_cast<bool>(packed_owner.qnk_packed));
        const std::shared_ptr<const QnKPack> packed_owner_sidecar = packed_owner.qnk_packed;
        ActivationBuffer owner_packed_projected_again;
        forward_linear(matrix,
                       input,
                       owner_packed_projected_again,
                       OptimizationCpuPackedWeights,
                       &packed_owner,
                       ExecutionBackend::Cpu);
        check(static_cast<bool>(packed_owner.qnk_packed == packed_owner_sidecar));
        check(packed_projected.rows() == input_rows);
        check(packed_projected.columns() == rows);
        check(owner_packed_projected.rows() == input_rows);
        check(owner_packed_projected.columns() == rows);
        check(owner_packed_projected_again.rows() == input_rows);
        check(owner_packed_projected_again.columns() == rows);

        if (dtype == DType::Q4K)
        {
            TensorData zero_matrix = matrix;
            std::fill(zero_matrix.quantized_data.begin(), zero_matrix.quantized_data.end(), uint8_t{0});
            CompiledOperatorTable empty_operators;
            const CompiledOperator* first_empty_owner = empty_operators.find_weight(0);
            const CompiledOperator* second_empty_owner = empty_operators.find_weight(1);
            check(first_empty_owner == nullptr);
            check(second_empty_owner == nullptr);
            check(&empty_operators.at_weight(0) == &empty_operators.at_weight(1));

            const ActivationBuffer matrix_without_owner = forward_linear(matrix, input, OptimizationCpuPackedWeights);
            const ActivationBuffer matrix_with_found_owner = forward_linear(matrix, input, OptimizationCpuPackedWeights, first_empty_owner);
            const ActivationBuffer zero_without_owner = forward_linear(zero_matrix, input, OptimizationCpuPackedWeights);
            const ActivationBuffer zero_with_found_owner = forward_linear(zero_matrix, input, OptimizationCpuPackedWeights, second_empty_owner);
            for (size_t token = 0; token < input_rows; ++token)
            {
                for (size_t row = 0; row < rows; ++row)
                {
                    check_near(matrix_with_found_owner.row(token)[row],
                               matrix_without_owner.row(token)[row],
                               0.1f);
                    check_near(zero_with_found_owner.row(token)[row], 0.0f, 0.1f);
                    check_near(zero_with_found_owner.row(token)[row],
                               zero_without_owner.row(token)[row],
                               0.1f);
                }
            }

            CompiledOperatorTable bound_operators;
            bound_operators.bind_weight_count(2);
            const CompiledOperator* first_bound_owner = bound_operators.find_weight(0);
            const CompiledOperator* second_bound_owner = bound_operators.find_weight(1);
            check(first_bound_owner != nullptr);
            check(second_bound_owner != nullptr);
            check(first_bound_owner != second_bound_owner);
            check(first_bound_owner == &bound_operators.at_weight(0));
            check(second_bound_owner == &bound_operators.at_weight(1));

            check(bound_operators.allocate() == 2);
            check(bound_operators.find_weight(2) == nullptr);
            bound_operators.bind_weight_count(4);
            check(bound_operators.size() == 4);
            check(bound_operators.find_weight(3) != nullptr);
            bound_operators.bind_weight_count(1);
            check(bound_operators.size() == 1);
            check(bound_operators.find_weight(0) != nullptr);
            check(bound_operators.find_weight(1) == nullptr);
            bound_operators.bind_weight_count(0);
            check(bound_operators.size() == 0);
            check(bound_operators.find_weight(0) == nullptr);
        }

        alignas(64) float decoded[qnk_block_elements];
        for (size_t token = 0; token < input_rows; ++token)
        {
            for (size_t row = 0; row < rows; ++row)
            {
                float expected = 0.0f;
                for (uint32_t block = 0; block < block_count; ++block)
                {
                    const uint8_t* encoded = raw.data() + (row * block_count + block) * block_bytes;
                    qnk_dequantize_block(dtype, encoded, decoded);
#if defined(__aarch64__) || defined(_M_ARM64)
                    // Fused unpacking must preserve the FP32 dot accumulation order.
                    const float* values = input.row(token) + block * qnk_block_elements;
                    const float dot = qnk_dot_block(dtype, encoded, values);
                    const float reference_dot = float_dot(decoded, values, qnk_block_elements);
                    check(std::bit_cast<uint32_t>(dot) == std::bit_cast<uint32_t>(reference_dot));
#endif
                    for (uint32_t column = 0; column < qnk_block_elements; ++column)
                        expected += decoded[column] * input.row(token)[block * qnk_block_elements + column];
                }
                check_near(projected.row(token)[row], expected, 0.1f);
                check_near(packed_projected.row(token)[row], expected, 0.1f);
                check_near(packed_projected.row(token)[row], projected.row(token)[row], 0.1f);
                check_near(owner_packed_projected.row(token)[row], packed_projected.row(token)[row], 0.1f);
            }
        }

        // Exercise decode, complete GEMM tiles and token/output tails with
        // reused output storage. The wider case also fills an OpenMP team.
        CpuOpenMpThreadLimitScope thread_limit;
        for (uint32_t threads : {1u, 2u, 8u})
        {
            thread_limit.set(threads);
            for (size_t tokens : {0u, 1u, 4u, 5u, 65u, 1u})
            {
                TensorData tiled = matrix;
                const uint32_t outputs = tokens == 65 ? 257u : static_cast<uint32_t>(rows);
                tiled.shape[0] = outputs;
                const size_t row_bytes = block_count * block_bytes;
                tiled.quantized_data.resize(static_cast<size_t>(outputs) * row_bytes);
                for (uint32_t row = 0; row < outputs; ++row)
                    std::copy_n(raw.data() + (row % rows) * row_bytes,
                                row_bytes, tiled.quantized_data.data() + static_cast<size_t>(row) * row_bytes);
                ActivationBuffer values(tokens, columns);
                for (size_t token = 0; token < tokens; ++token)
                    for (uint32_t column = 0; column < columns; ++column)
                        values.row(token)[column] = static_cast<float>(static_cast<int>((column * 7 + token * 3) % 29) - 14) * 0.015625f;
                CompiledOperator owner;
                for (uint64_t flags : {uint64_t{0}, uint64_t{OptimizationCpuPackedWeights}})
                {
#if defined(_OPENMP)
                    if (threads == 2 && tokens == 65)
                    {
#pragma omp parallel num_threads(2)
                        {
#pragma omp single
                            forward_linear(tiled, values, projected, flags, &owner, ExecutionBackend::Cpu);
                        }
                    }
                    else
#endif
                        forward_linear(tiled, values, projected, flags, &owner, ExecutionBackend::Cpu);
                    check(projected.rows() == tokens && projected.columns() == outputs);
                    for (uint32_t row = 0; row < outputs; ++row)
                    {
                        std::array<float, columns> reference{};
                        for (uint32_t block = 0; block < block_count; ++block)
                            qnk_dequantize_block(dtype, raw.data() + ((row % rows) * block_count + block) * block_bytes,
                                                 reference.data() + block * qnk_block_elements);
                        for (size_t token = 0; token < tokens; ++token)
                        {
                            double expected = 0.0;
                            for (uint32_t column = 0; column < columns; ++column)
                                expected += static_cast<double>(reference[column]) * values.row(token)[column];
                            check_near(projected.row(token)[row], static_cast<float>(expected),
                                       3e-5f * std::max(1.0f, std::fabs(static_cast<float>(expected))));
                        }
                    }
                }
            }
        }
    }
}

void test_qnk_graph_cpu_projections()
{
    constexpr uint32_t vocabulary_size = 256;
    constexpr uint32_t hidden_size = 256;
    constexpr uint32_t intermediate_size = 256;
    constexpr uint32_t expert_count = 2;
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };

    for (const DType dtype : dtypes)
    {
        MoeModelDescriptor descriptor;
        descriptor.model_type = "qnk_graph_fusion_test";
        descriptor.vocabulary_size = vocabulary_size;
        descriptor.hidden_size = hidden_size;
        descriptor.intermediate_size = intermediate_size;
        descriptor.expert_count = expert_count;
        descriptor.experts_per_token = 1;
        descriptor.activation_dtype = DType::Float32;
        descriptor.kv_cache_dtype = DType::Float32;
        descriptor.norm_epsilon = 1e-5f;
        descriptor.layers.resize(1);
        descriptor.layers.front().pre_ffn_norm = NormType::RmsNorm;
        MoeDescriptor& moe = descriptor.layers.front().moe;
        moe.expert_count = expert_count;
        moe.top_k = 1;
        moe.intermediate_size = intermediate_size;
        moe.activation = ExpertActivation::Silu;
        moe.layout = ExpertLayout::GateUpDown;
        moe.expert_weight_dtype = dtype;

        WeightMapping mapping;
        auto add_float = [&mapping](const std::string& name, std::vector<uint32_t> shape) {
            TensorData tensor;
            tensor.dtype = DType::Float32;
            tensor.shape = std::move(shape);
            tensor.float32_data.assign(static_cast<size_t>(tensor.element_count()), 0.0f);
            mapping.emplace(name, std::move(tensor));
        };
        auto add_qnk = [&mapping, dtype](const std::string& name, uint32_t rows, uint32_t columns) {
            TensorData tensor;
            tensor.dtype = dtype;
            tensor.shape = {rows, columns};
            tensor.quantized_data.resize(static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns)), 0);
            mapping.emplace(name, std::move(tensor));
        };

        add_float("token_embedding.weight", {vocabulary_size, hidden_size});
        add_float("final_norm.weight", {hidden_size});
        add_float("lm_head.weight", {vocabulary_size, hidden_size});
        add_float("layers.0.pre_ffn_norm.weight", {hidden_size});
        add_float("layers.0.router.weight", {expert_count, hidden_size});
        for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id)
        {
            const std::string prefix = "layers.0.experts." + std::to_string(expert_id) + ".";
            add_qnk(prefix + "gate.weight", intermediate_size, hidden_size);
            add_qnk(prefix + "up.weight", intermediate_size, hidden_size);
            add_qnk(prefix + "down.weight", hidden_size, intermediate_size);
        }

        CompilerOption compiler_opt;
        compiler_opt.flags |= BackendVulkanDense;
        compiler_opt.optimization_flags = OptimizationDefaultFlags;
        compiler_opt.num_concurrent_sessions = 3;
        const bool test_hybrid_plan = dtype == DType::Q4K && get_gpu_count() != 0;
        std::unique_ptr<ScopedTestDirectory> hybrid_directory;
        MoeModelDescriptor hybrid_descriptor;
        WeightMapping hybrid_mapping;
        struct MappedQnkTensor
        {
            uint32_t expert_id = 0;
            uint32_t projection = 0;
            std::string name;
            size_t size = 0;
            std::shared_ptr<const uint8_t> data;
        };
        std::vector<MappedQnkTensor> mapped_qnk;
        const std::array<const char*, 3> projections = {"gate", "up", "down"};
        if (test_hybrid_plan)
        {
            hybrid_descriptor = descriptor;
            hybrid_mapping = mapping;
            hybrid_directory = std::make_unique<ScopedTestDirectory>("ncnn_moe_qnk_hybrid_mapping_test_");
            const std::filesystem::path bank_path = hybrid_directory->path() / "experts.bin";
            {
                std::ofstream bank(bank_path, std::ios::binary);
                for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id)
                {
                    const std::string prefix = "layers.0.experts." + std::to_string(expert_id) + ".";
                    for (uint32_t projection = 0; projection < projections.size(); ++projection)
                    {
                        const std::string name = prefix + projections[projection] + ".weight";
                        TensorData& weight = hybrid_mapping.at(name);
                        const std::span<const uint8_t> values = weight.qnk_values();
                        check(!values.empty());
                        bank.write(reinterpret_cast<const char*>(values.data()),
                                   static_cast<std::streamsize>(values.size()));
                        check(static_cast<bool>(bank));
                        mapped_qnk.push_back({expert_id, projection, name, values.size(), {}});
                    }
                }
                bank.close();
                check(static_cast<bool>(bank));
            }

            size_t offset = 0;
            for (MappedQnkTensor& source : mapped_qnk)
            {
                TensorData& weight = hybrid_mapping.at(source.name);
                auto range = MappedFileRange::open(bank_path, offset, source.size);
                check(static_cast<bool>(range));
                source.data = range.value()->share_data();
                check(static_cast<bool>(source.data));
                weight.mapped_size = source.size;
                weight.mapped_data = source.data;
                std::vector<uint8_t>().swap(weight.quantized_data);
                offset += source.size;
            }
        }
        CompiledModel model;
        auto ret = compile_model(std::move(descriptor), std::move(mapping), model, HybridMode::CpuOnly, compiler_opt);
        check(static_cast<bool>(ret));
        check(model.opt.hybrid_mode == HybridMode::CpuOnly);
        check(model.opt.vulkan_device_index == automatic_vulkan_device_index);
        check(model.opt.vulkan_device_indices.empty());
        check(model.opt.flags == 0);
        check(model.opt.optimization_flags == compiler_opt.optimization_flags);
        check(model.opt.num_concurrent_sessions == compiler_opt.num_concurrent_sessions);
        check(static_cast<bool>(model.net.layer_plans.size() == 1));
        const ExpertPlan& expert = model.net.layer_plans.front().moe.experts.front();
        check(expert.gate_up_weight == invalid_tensor_handle);
        check(expert.gate_weight != invalid_tensor_handle);
        check(expert.up_weight != invalid_tensor_handle);
        check(expert.layout == ExpertLayout::GateUpDown);
        for (TensorHandle handle : {expert.gate_weight, expert.up_weight})
        {
            const TensorData& weight = model.weights.at(handle);
            check(weight.dtype == dtype);
            check((weight.shape == std::vector<uint32_t>{intermediate_size, hidden_size}));
            check(weight.qnk_values().size() == qnk_storage_bytes(dtype, intermediate_size, hidden_size));
        }

        if (test_hybrid_plan)
        {
            CompilerOption hybrid_opt = compiler_opt;
            hybrid_opt.flags |= BackendVulkanExperts;
            hybrid_opt.device_index = get_default_gpu_index();
            hybrid_opt.vulkan_runtime = create_vulkan_runtime();
            CompiledModel hybrid_model;
            auto hybrid_ret = compile_model(std::move(hybrid_descriptor),
                                            std::move(hybrid_mapping),
                                            hybrid_model,
                                            HybridMode::HybridExperts,
                                            hybrid_opt);
            if (!hybrid_ret)
                throw std::runtime_error("Qn_K hybrid graph compilation failed: " + hybrid_ret.error().message);
            const ExpertPlan& hybrid_expert = hybrid_model.net.layer_plans.front().moe.experts.front();
            check(hybrid_expert.layout == ExpertLayout::GateUpDown);
            check(hybrid_expert.gate_up_weight == invalid_tensor_handle);
            check(hybrid_expert.gate_weight != invalid_tensor_handle);
            check(hybrid_expert.up_weight != invalid_tensor_handle);
            for (const MappedQnkTensor& source : mapped_qnk)
            {
                const ExpertPlan& hybrid_plan = hybrid_model.net.layer_plans.front().moe.experts[source.expert_id];
                const ExpertPlan& cpu_plan = model.net.layer_plans.front().moe.experts[source.expert_id];
                const std::array<TensorHandle, 3> hybrid_handles = {
                    hybrid_plan.gate_weight,
                    hybrid_plan.up_weight,
                    hybrid_plan.down_weight,
                };
                const std::array<TensorHandle, 3> cpu_handles = {
                    cpu_plan.gate_weight,
                    cpu_plan.up_weight,
                    cpu_plan.down_weight,
                };
                const TensorData& hybrid_weight = hybrid_model.weights.at(hybrid_handles[source.projection]);
                const TensorData& cpu_weight = model.weights.at(cpu_handles[source.projection]);
                const auto hybrid_values = hybrid_weight.qnk_values();
                const auto cpu_values = cpu_weight.qnk_values();
                check(hybrid_weight.mapped_data.get() == source.data.get());
                check(hybrid_weight.mapped_size == source.size);
                check(hybrid_weight.quantized_data.empty());
                check(hybrid_values.data() == source.data.get());
                check(hybrid_values.size() == source.size);
                check(hybrid_values.size() == cpu_values.size());
                check(std::equal(hybrid_values.begin(), hybrid_values.end(), cpu_values.begin()));
            }
            hybrid_model = {};
            hybrid_mapping.clear();
            mapped_qnk.clear();
        }

        WeightStore capability_weights;
        TensorData gate_up = model.weights.at(expert.gate_weight);
        gate_up.shape[0] *= 2;
        gate_up.quantized_data.resize(static_cast<size_t>(qnk_storage_bytes(dtype, intermediate_size * 2, hidden_size)));
        auto gate_up_handle = capability_weights.add("gate_up", std::move(gate_up));
        auto down_handle = capability_weights.add("down", model.weights.at(expert.down_weight));
        check(static_cast<bool>(gate_up_handle) && static_cast<bool>(down_handle));
        ExpertPlan packed_expert = expert;
        packed_expert.gate_up_weight = gate_up_handle.value();
        packed_expert.down_weight = down_handle.value();
        MoeBlockPlan capability_moe;
        capability_moe.experts.push_back(packed_expert);
        const auto eligible = [&] {
            return support_vulkan_expert(packed_expert, capability_weights.at(gate_up_handle.value()),
                                         capability_weights.at(down_handle.value()), compiler_opt.optimization_flags);
        };
        check(eligible());
        check(support_vulkan_experts(capability_weights, capability_moe, compiler_opt.optimization_flags));
        for (TensorHandle handle : {gate_up_handle.value(), down_handle.value()})
        {
            TensorData& weight = capability_weights.at_mutable(handle);
            weight.input_column_indices.resize(weight.shape[1]);
            for (uint32_t column = 0; column < weight.shape[1]; ++column)
                weight.input_column_indices[column] = weight.shape[1] - column - 1;
            check(eligible());
            check(support_vulkan_experts(capability_weights, capability_moe, compiler_opt.optimization_flags));
            weight.input_column_indices.clear();
        }
    }
}

} // namespace moe
} // namespace ncnn
