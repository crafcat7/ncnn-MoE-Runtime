#include "kernels/ops.h"
#include "kernels/qnk.h"
#include "graph/compiledoperator.h"
#include "ncnn/moe/option.h"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ncnn {
namespace moe {

static void q8_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void q8_near(float actual, double expected, float tolerance = 2e-5f)
{
    q8_check(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance * std::max(1.0, std::fabs(expected)), "Q8_0 numerical mismatch");
}

static TensorData make_q8_0(uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData tensor;
    tensor.dtype = DType::Q8_0;
    tensor.shape = {rows, columns};
    tensor.quantized_data.resize(static_cast<size_t>(rows) * (columns / 32) * 34);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t block = 0; block < columns / 32; ++block)
        {
            uint8_t* encoded = tensor.quantized_data.data() + (static_cast<size_t>(row) * (columns / 32) + block) * 34;
            const uint16_t scale = (row + block + seed) % 2 == 0 ? 0x2400 : 0xa800;
            std::memcpy(encoded, &scale, sizeof(scale));
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const int value = static_cast<int>((row * 37 + block * 23 + lane * 17 + seed * 11) % 256) - 128;
                encoded[2 + lane] = static_cast<uint8_t>(value);
            }
            encoded[2] = 0x80;
            encoded[3] = 0x7f;
        }
    return tensor;
}

static double q8_reference(const TensorData& matrix, uint32_t row, const float* input)
{
    const uint32_t columns = matrix.shape[1];
    const uint8_t* weights = matrix.qnk_values().data() + static_cast<size_t>(row) * (columns / 32) * 34;
    double sum = 0.0;
    for (uint32_t block = 0; block < columns / 32; ++block)
    {
        const uint8_t* encoded = weights + block * 34;
        uint16_t bits;
        std::memcpy(&bits, encoded, sizeof(bits));
        // The fixtures use exact powers of two. This oracle is independent
        // of the native half decoder and signed-byte dot implementation.
        const double scale = bits == 0x2400 ? 0.015625 : -0.03125;
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const int value = encoded[2 + lane] < 128 ? encoded[2 + lane] : static_cast<int>(encoded[2 + lane]) - 256;
            sum += scale * value * input[block * 32 + lane];
        }
    }
    return sum;
}

static void test_block_decoder()
{
    std::array<uint8_t, 36> storage{};
    uint8_t* encoded = storage.data() + 1;
    std::array<float, 33> decoded{};
    std::array<float, 32> input{};
    const std::array<uint16_t, 6> bits{0x3c00, 0xbc00, 0x0001, 0x8001, 0x0000, 0x8000};
    const std::array<float, 6> scales{1.0f, -1.0f, 0x1p-24f, -0x1p-24f, 0.0f, -0.0f};
    for (size_t scale_index = 0; scale_index < bits.size(); ++scale_index)
    {
        std::memcpy(encoded, &bits[scale_index], sizeof(uint16_t));
        decoded[32] = 12345.0f;
        double dot = 0.0;
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const int value = lane == 0 ? -128 : lane == 1 ? 127
                                                           : static_cast<int>(lane) - 16;
            encoded[2 + lane] = static_cast<uint8_t>(value);
            input[lane] = static_cast<float>(static_cast<int>(lane % 11) - 5) * 0.125f;
            dot += scales[scale_index] * value * input[lane];
        }
        qnk_dequantize_block(DType::Q8_0, encoded, decoded.data());
        q8_check(decoded[32] == 12345.0f, "Q8_0 decoder wrote past 32 values");
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const int value = encoded[2 + lane] < 128 ? encoded[2 + lane] : static_cast<int>(encoded[2 + lane]) - 256;
            q8_check(decoded[lane] == scales[scale_index] * value, "Q8_0 decoder signed scale or code");
        }
        const float observed = qnk_dot_block(DType::Q8_0, encoded, input.data());
        q8_check(std::isfinite(observed) && std::fabs(observed - dot) <= std::max(1e-30, std::fabs(dot) * 1e-6), "Q8_0 dot subnormal scale or signed code");
    }
}

static void test_linear_and_embedding()
{
    for (uint32_t columns : {32u, 64u, 96u, 256u, 288u})
        for (uint32_t rows : {1u, 9u})
        {
            TensorData matrix = make_q8_0(rows, columns, 1);
            const std::vector<uint8_t> original = matrix.quantized_data;
            // All consumers must accept file ranges with no 16/32-bit alignment.
            auto storage = std::make_shared<std::vector<uint8_t>>(original.size() + 1);
            std::memcpy(storage->data() + 1, original.data(), original.size());
            matrix.mapped_data = std::shared_ptr<const uint8_t>(storage, storage->data() + 1);
            matrix.mapped_size = original.size();
            matrix.quantized_data.clear();
            q8_check(matrix.qnk_values().size() == original.size(), "Q8_0 mapped view");
            QnKPack pack;
            q8_check(qnk_pack_weights(original.data(), original.size(), DType::Q8_0, rows, columns, pack), "Q8_0 pack");
            q8_check(pack.block_count == columns / 32 && pack.storage.size() == qnk_packed_storage_bytes(DType::Q8_0, rows, columns), "Q8_0 packed dimensions");
            for (uint32_t row = 0; row < rows; ++row)
                for (uint32_t block = 0; block < columns / 32; ++block)
                    q8_check(std::memcmp(qnk_packed_block(pack, row, block), original.data() + (static_cast<size_t>(row) * (columns / 32) + block) * 34, 34) == 0, "Q8_0 packing changed native bytes");
            for (size_t tokens : {size_t{0}, size_t{1}, size_t{4}, size_t{5}})
            {
                ActivationBuffer input(tokens, columns);
                ActivationBuffer bf_input(tokens, columns, DType::BFloat16);
                for (size_t token = 0; token < tokens; ++token)
                    for (uint32_t column = 0; column < columns; ++column)
                    {
                        const float value = static_cast<float>(static_cast<int>((token * 7 + column * 3) % 29) - 14) * 0.0625f;
                        input.row(token)[column] = value;
                        const uint16_t bf = float_to_bfloat16(value);
                        std::memcpy(bf_input.mutable_row_bytes(token).data() + column * sizeof(bf), &bf, sizeof(bf));
                    }
                for (uint64_t flags : {uint64_t{0}, uint64_t{OptimizationCpuPackedWeights}})
                {
                    CompiledOperator owner;
                    ActivationBuffer output;
                    forward_linear(matrix, input, output, flags, &owner, ExecutionBackend::Cpu);
                    q8_check(output.rows() == tokens && output.columns() == rows, "Q8_0 linear shape");
                    ActivationBuffer bf_output(0, 0, DType::BFloat16);
                    ActivationBuffer bf_float_output;
                    forward_linear(matrix, bf_input, bf_output, flags, &owner, ExecutionBackend::Cpu);
                    forward_linear(matrix, bf_input, bf_float_output, flags, &owner, ExecutionBackend::Cpu);
                    ActivationBuffer float_bf_output(0, 0, DType::BFloat16);
                    forward_linear(matrix, input, float_bf_output, flags, &owner, ExecutionBackend::Cpu);
                    for (size_t token = 0; token < tokens; ++token)
                        for (uint32_t row = 0; row < rows; ++row)
                        {
                            const double reference = q8_reference(matrix, row, input.row(token));
                            q8_near(output.row(token)[row], reference);
                            q8_near(bf_float_output.row(token)[row], reference);
                            uint16_t bf, from_float;
                            std::memcpy(&bf, bf_output.row_bytes(token).data() + row * sizeof(bf), sizeof(bf));
                            std::memcpy(&from_float, float_bf_output.row_bytes(token).data() + row * sizeof(from_float), sizeof(from_float));
                            q8_near(bfloat16_to_float(bf), reference, 0.004f);
                            q8_check(bf == from_float, "Q8_0 BF16 activation parity");
                        }
                    if (tokens != 0 && flags != 0)
                        q8_check(owner.qnk_packed && owner.qnk_packed->dtype == DType::Q8_0, "Q8_0 persistent packed sidecar");
                }
            }
            std::vector<float> decoded(columns);
            std::vector<float> remapped(columns);
            for (uint32_t row = 0; row < rows; ++row)
            {
                forward_embedding_row(matrix, row, decoded.data());
                TensorData permuted = matrix;
                permuted.input_column_indices.resize(columns);
                for (uint32_t column = 0; column < columns; ++column)
                    permuted.input_column_indices[column] = columns - 1 - column;
                forward_embedding_row(permuted, row, remapped.data());
                for (uint32_t column = 0; column < columns; ++column)
                    q8_check(remapped[columns - 1 - column] == decoded[column], "Q8_0 embedding column map");
            }
            q8_check(std::memcmp(matrix.qnk_values().data(), original.data(), original.size()) == 0, "Q8_0 execution mutated native weights");
        }
}

static void test_paired_and_rejections()
{
    const TensorData first = make_q8_0(9, 96, 1);
    const TensorData second = make_q8_0(3, 96, 2);
    ActivationBuffer input(5, 96);
    for (size_t token = 0; token < input.rows(); ++token)
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(token)[column] = static_cast<float>(static_cast<int>((column + token * 5) % 23) - 11) * 0.125f;
    for (uint64_t flags : {uint64_t{0}, uint64_t{OptimizationCpuPackedWeights}})
    {
        ActivationBuffer left, right;
        q8_check(forward_linear({&first, &second}, input, {&left, &right}, flags), "Q8_0 paired projection");
        ActivationBuffer bf_input(input.rows(), input.columns(), DType::BFloat16);
        for (size_t token = 0; token < input.rows(); ++token)
            for (uint32_t column = 0; column < input.columns(); ++column)
            {
                const uint16_t value = float_to_bfloat16(input.row(token)[column]);
                std::memcpy(bf_input.mutable_row_bytes(token).data() + column * sizeof(value), &value, sizeof(value));
            }
        ActivationBuffer bf_left(0, 0, DType::BFloat16);
        ActivationBuffer bf_right;
        q8_check(forward_linear_qnk({&first, &second}, bf_input, {&bf_left, &bf_right}, flags != 0), "Q8_0 BF16 paired projection");
        for (size_t token = 0; token < input.rows(); ++token)
        {
            for (uint32_t row = 0; row < first.shape[0]; ++row)
            {
                q8_near(left.row(token)[row], q8_reference(first, row, input.row(token)));
                uint16_t value;
                std::memcpy(&value, bf_left.row_bytes(token).data() + row * sizeof(value), sizeof(value));
                q8_near(bfloat16_to_float(value), q8_reference(first, row, input.row(token)), 0.004f);
            }
            for (uint32_t row = 0; row < second.shape[0]; ++row)
            {
                q8_near(right.row(token)[row], q8_reference(second, row, input.row(token)));
                q8_near(bf_right.row(token)[row], q8_reference(second, row, input.row(token)));
            }
        }
    }
    ActivationBuffer output;
    TensorData truncated = first;
    truncated.quantized_data.pop_back();
    q8_check(!forward_linear_qnk({&truncated}, input, {&output}, false), "Q8_0 truncated input accepted");
    q8_check(!forward_linear_qnk({&first}, input, {&input}, false), "Q8_0 aliased output accepted");
    q8_check(!forward_linear_qnk({&first, &second}, input, {&output, &output}, true), "Q8_0 aliased pair accepted");
    for (uint32_t columns : {0u, 31u, 33u, 255u})
        q8_check(!qnk_shape_supported(DType::Q8_0, 1, columns) && qnk_storage_bytes(DType::Q8_0, 1, columns) == 0, "Q8_0 invalid block width");
    q8_check(qnk_block_size(DType::Q8_0) == 32 && qnk_block_bytes(DType::Q8_0) == 34, "Q8_0 ABI");
    q8_check(qnk_storage_bytes(DType::Q8_0, std::numeric_limits<size_t>::max(), std::numeric_limits<uint32_t>::max() - 31) == 0, "Q8_0 byte-size overflow");
    for (DType dtype : {DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8K})
        q8_check(qnk_block_size(dtype) == 256 && !qnk_shape_supported(dtype, 1, 32), "Qn_K block contract changed");
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        ncnn::moe::test_block_decoder();
        ncnn::moe::test_linear_and_embedding();
        ncnn::moe::test_paired_and_rejections();
        std::cout << "native Q8_0 CPU tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
