#include "graph/compiledoperator.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"
#include "ncnn/moe/option.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ncnn {
namespace moe {

static void block_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void block_near(float actual, double expected, double tolerance = 3e-5)
{
    block_check(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance * std::max(1.0, std::fabs(expected)),
                "native GGUF block numerical mismatch");
}

static double reference_half(const uint8_t* source)
{
    const uint16_t bits = static_cast<uint16_t>(source[0] | (static_cast<uint16_t>(source[1]) << 8));
    const unsigned exponent = (bits >> 10) & 31u;
    const unsigned mantissa = bits & 1023u;
    const double magnitude = exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -24)
                                           : std::ldexp(static_cast<double>(1024 + mantissa), static_cast<int>(exponent) - 25);
    return (bits & 0x8000u) ? -magnitude : magnitude;
}

static void write_half(uint8_t* output, uint16_t bits)
{
    output[0] = static_cast<uint8_t>(bits);
    output[1] = static_cast<uint8_t>(bits >> 8);
}

static uint32_t reference_block_size(DType dtype)
{
    return dtype == DType::IQ4XS ? 256 : 32;
}

static size_t reference_block_bytes(DType dtype)
{
    switch (dtype)
    {
    case DType::Q4_0: return 18;
    case DType::Q4_1: return 20;
    case DType::Q5_0: return 22;
    case DType::IQ4NL: return 18;
    case DType::Q5_1: return 24;
    case DType::IQ4XS: return 136;
    default: return 0;
    }
}

static float reference_value(DType dtype, const uint8_t* source, uint32_t lane)
{
    // The oracle uses the published GGML byte layout, independent of runtime
    // half conversion, block widths, unpack functions and packed sidecars.
    static constexpr std::array<int, 16> nonlinear = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const double scale = reference_half(source);
    if (dtype == DType::Q4_0 || dtype == DType::Q4_1)
    {
        const bool affine = dtype == DType::Q4_1;
        const uint8_t* codes = source + (affine ? 4 : 2);
        const unsigned nibble = (codes[lane % 16] >> (4 * (lane / 16))) & 15u;
        const int value = static_cast<int>(nibble) - (affine ? 0 : 8);
        return static_cast<float>(scale * value + (affine ? reference_half(source + 2) : 0.0));
    }
    if (dtype == DType::Q5_0)
    {
        const uint32_t high_bits = static_cast<uint32_t>(source[2])
                                   | (static_cast<uint32_t>(source[3]) << 8)
                                   | (static_cast<uint32_t>(source[4]) << 16)
                                   | (static_cast<uint32_t>(source[5]) << 24);
        const unsigned nibble = (source[6 + lane % 16] >> (4 * (lane / 16))) & 15u;
        const int value = static_cast<int>(nibble | (((high_bits >> lane) & 1u) << 4)) - 16;
        return static_cast<float>(scale * value);
    }
    if (dtype == DType::IQ4XS)
    {
        const uint16_t scales_high = static_cast<uint16_t>(source[2] | (static_cast<uint16_t>(source[3]) << 8));
        const uint32_t group = lane / 32;
        const uint32_t within = lane % 32;
        const uint32_t packed_scale = ((source[4 + group / 2] >> ((group & 1u) * 4)) & 15u)
                                      | (((scales_high >> (group * 2)) & 3u) << 4);
        const unsigned nibble = (source[8 + group * 16 + within % 16] >> (4 * (within / 16))) & 15u;
        return static_cast<float>(scale * (static_cast<int>(packed_scale) - 32) * nonlinear[nibble]);
    }
    if (dtype == DType::IQ4NL)
    {
        const unsigned nibble = (source[2 + lane % 16] >> (4 * (lane / 16))) & 15u;
        return static_cast<float>(scale * nonlinear[nibble]);
    }
    const unsigned low = (source[8 + lane % 16] >> (4 * (lane / 16))) & 15u;
    const unsigned high = ((source[4 + lane / 8] >> (lane % 8)) & 1u) * 16;
    return static_cast<float>(scale * (low + high) + reference_half(source + 2));
}

static TensorData make_blocks(DType dtype, uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData tensor;
    tensor.dtype = dtype;
    tensor.shape = {rows, columns};
    const uint32_t block_size = reference_block_size(dtype);
    const size_t block_bytes = reference_block_bytes(dtype);
    tensor.quantized_data.resize(static_cast<size_t>(rows) * (columns / block_size) * block_bytes);
    static constexpr std::array<uint16_t, 8> scales = {0x2400, 0xa800, 0x1800, 0x9400, 0x0001, 0x8001, 0, 0x8000};
    static constexpr std::array<uint16_t, 5> minima = {0x3000, 0xb000, 0, 0x0001, 0x8001};
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t block = 0; block < columns / block_size; ++block)
        {
            uint8_t* encoded = tensor.quantized_data.data() + (static_cast<size_t>(row) * (columns / block_size) + block) * block_bytes;
            write_half(encoded, scales[(row + block + seed) % scales.size()]);
            const size_t code_offset = dtype == DType::IQ4NL   ? 2
                                       : dtype == DType::Q4_0  ? 2
                                       : dtype == DType::Q4_1  ? 4
                                       : dtype == DType::Q5_0  ? 6
                                       : dtype == DType::IQ4XS ? 8
                                                               : 8;
            if (dtype == DType::Q4_1)
                write_half(encoded + 2, minima[(row * 3 + block + seed) % minima.size()]);
            if (dtype == DType::Q5_0)
                for (uint32_t byte = 0; byte < 4; ++byte)
                    encoded[2 + byte] = static_cast<uint8_t>(row * 97 + block * 43 + seed * 19 + byte * 53);
            if (dtype == DType::IQ4XS)
            {
                encoded[2] = static_cast<uint8_t>(row * 13 + block * 17 + seed * 19);
                encoded[3] = static_cast<uint8_t>(row * 37 + block * 29 + seed * 31);
                for (uint32_t byte = 0; byte < 4; ++byte)
                    encoded[4 + byte] = static_cast<uint8_t>(row * 71 + block * 29 + seed * 13 + byte * 17);
            }
            if (dtype == DType::Q5_1)
            {
                write_half(encoded + 2, minima[(row * 3 + block + seed) % minima.size()]);
                for (uint32_t byte = 0; byte < 4; ++byte)
                    encoded[4 + byte] = static_cast<uint8_t>(row * 97 + block * 43 + seed * 19 + byte * 53);
            }
            for (uint32_t byte = 0; byte < block_bytes - code_offset; ++byte)
                encoded[code_offset + byte] = static_cast<uint8_t>(row * 71 + block * 29 + seed * 13 + byte * 17);
        }
    return tensor;
}

static double reference_dot(const TensorData& matrix, uint32_t row, const float* input)
{
    const uint32_t columns = matrix.shape[1];
    const uint32_t block_size = reference_block_size(matrix.dtype);
    const size_t bytes = reference_block_bytes(matrix.dtype);
    const uint8_t* source = matrix.qnk_values().data() + static_cast<size_t>(row) * (columns / block_size) * bytes;
    double sum = 0.0;
    for (uint32_t column = 0; column < columns; ++column)
        sum += static_cast<double>(reference_value(matrix.dtype, source + (column / block_size) * bytes, column % block_size)) * input[column];
    return sum;
}

static void test_native_block_decoders()
{
    std::array<uint8_t, 137> storage{};
    uint8_t* source = storage.data() + 1;
    std::array<float, 257> output;
    std::array<float, 256> input;
    static constexpr std::array<uint16_t, 8> scales = {0x3c00, 0xbc00, 0x0001, 0x8001, 0x03ff, 0x83ff, 0, 0x8000};
    static constexpr std::array<uint16_t, 5> minima = {0, 0x3c00, 0xbc00, 0x0001, 0x8001};
    for (uint32_t lane = 0; lane < input.size(); ++lane)
        input[lane] = static_cast<float>(static_cast<int>((lane * 7) % 19) - 9) * 0.125f;
    for (DType dtype : {DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::IQ4NL, DType::Q5_1, DType::IQ4XS})
        for (uint16_t scale : scales)
            for (size_t minimum_index = 0; minimum_index < (dtype == DType::Q4_1 || dtype == DType::Q5_1 ? minima.size() : 1); ++minimum_index)
                for (uint32_t pattern = 0; pattern < 256; ++pattern)
                {
                    write_half(source, scale);
                    const uint32_t block_size = reference_block_size(dtype);
                    const size_t block_bytes = reference_block_bytes(dtype);
                    const size_t offset = dtype == DType::IQ4NL || dtype == DType::Q4_0 ? 2
                                          : dtype == DType::Q4_1                        ? 4
                                          : dtype == DType::Q5_0                        ? 6
                                          : dtype == DType::IQ4XS                       ? 8
                                                                                        : 8;
                    if (dtype == DType::Q4_1 || dtype == DType::Q5_1)
                    {
                        write_half(source + 2, minima[minimum_index]);
                    }
                    if (dtype == DType::Q5_0 || dtype == DType::Q5_1)
                    {
                        // Exercise every high-bit position, plus all-clear,
                        // all-set and alternating 32-bit patterns.
                        const uint32_t choice = pattern % 36;
                        const uint32_t high = choice < 32    ? (UINT32_C(1) << choice)
                                              : choice == 32 ? 0u
                                              : choice == 33 ? UINT32_MAX
                                              : choice == 34 ? 0x55555555u
                                                             : 0xaaaaaaaau;
                        const uint32_t high_offset = dtype == DType::Q5_0 ? 2 : 4;
                        for (uint32_t byte = 0; byte < 4; ++byte)
                            source[high_offset + byte] = static_cast<uint8_t>(high >> (byte * 8));
                    }
                    if (dtype == DType::IQ4XS)
                    {
                        source[2] = static_cast<uint8_t>(pattern * 13u);
                        source[3] = static_cast<uint8_t>(pattern * 71u);
                        for (uint32_t byte = 0; byte < 4; ++byte)
                            source[4 + byte] = static_cast<uint8_t>(pattern + byte * 53u);
                    }
                    for (uint32_t byte = 0; byte < block_bytes - offset; ++byte)
                        source[offset + byte] = static_cast<uint8_t>(pattern + byte * 13);
                    output.fill(98765.0f);
                    qnk_dequantize_block(dtype, source, output.data());
                    block_check(output[block_size] == 98765.0f, "native decoder wrote beyond its GGUF block");
                    double expected_dot = 0.0;
                    double absolute_products = 0.0;
                    for (uint32_t lane = 0; lane < block_size; ++lane)
                    {
                        const float expected = reference_value(dtype, source, lane);
                        block_check(output[lane] == expected, "native GGUF scale/minimum/code layout mismatch");
                        expected_dot += static_cast<double>(expected) * input[lane];
                        absolute_products += std::fabs(static_cast<double>(expected) * input[lane]);
                    }
                    const float actual = qnk_dot_block(dtype, source, input.data());
                    // A small residual can follow nearly complete cancellation.
                    // Bound the FP32 products and additions by gamma(2n)
                    // times their absolute sum, rather than the residual alone.
                    // Exact per-lane decoding is checked independently above.
                    constexpr double rounding_unit = std::numeric_limits<float>::epsilon() / 2.0;
                    const double rounded_operations = 2.0 * block_size;
                    const double gamma = rounded_operations * rounding_unit / (1.0 - rounded_operations * rounding_unit);
                    const double tolerance = gamma * absolute_products
                                             + rounded_operations * std::numeric_limits<float>::denorm_min();
                    if (!std::isfinite(actual) || std::fabs(actual - expected_dot) > tolerance)
                    {
                        std::ostringstream diagnostic;
                        diagnostic << std::setprecision(17) << "native dot mismatch: dtype=" << static_cast<int>(dtype)
                                   << " scale_bits=" << scale << " minimum_bits=" << minima[minimum_index]
                                   << " pattern=" << pattern << " actual=" << actual << " expected=" << expected_dot
                                   << " absolute_error=" << std::fabs(actual - expected_dot) << " tolerance=" << tolerance
                                   << " absolute_products=" << absolute_products;
                        throw std::runtime_error(diagnostic.str());
                    }
                }
}

static void test_native_linear_and_embedding()
{
    for (DType dtype : {DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::IQ4NL, DType::Q5_1, DType::IQ4XS})
    {
        const uint32_t block_size = reference_block_size(dtype);
        const auto column_sizes = block_size == 32
                                      ? std::array<uint32_t, 5>{32u, 64u, 160u, 256u, 288u}
                                      : std::array<uint32_t, 5>{256u, 512u, 768u, 1024u, 1280u};
        for (uint32_t columns : column_sizes)
            for (uint32_t rows : {1u, 9u})
            {
                TensorData matrix = make_blocks(dtype, rows, columns, 1);
                const std::vector<uint8_t> original = matrix.quantized_data;
                const size_t bytes = reference_block_bytes(dtype);
                auto storage = std::make_shared<std::vector<uint8_t>>(original.size() + 1);
                std::memcpy(storage->data() + 1, original.data(), original.size());
                matrix.mapped_data = std::shared_ptr<const uint8_t>(storage, storage->data() + 1);
                matrix.mapped_size = original.size();
                matrix.quantized_data.clear();
                QnKPack pack;
                block_check(qnk_pack_weights(matrix.qnk_values().data(), original.size(), dtype, rows, columns, pack), "native raw-block pack failed");
                block_check(pack.block_count == columns / block_size && pack.storage.size() == qnk_packed_storage_bytes(dtype, rows, columns), "native packed shape mismatch");
                for (uint32_t row = 0; row < rows; ++row)
                    for (uint32_t block = 0; block < columns / block_size; ++block)
                        block_check(std::memcmp(qnk_packed_block(pack, row, block), original.data() + (static_cast<size_t>(row) * (columns / block_size) + block) * bytes, bytes) == 0,
                                    "packing changed the native GGUF block bytes");
                for (size_t tokens : {size_t{0}, size_t{1}, size_t{4}, size_t{5}})
                {
                    ActivationBuffer input(tokens, columns);
                    ActivationBuffer bf_input(tokens, columns, DType::BFloat16);
                    for (size_t token = 0; token < tokens; ++token)
                        for (uint32_t column = 0; column < columns; ++column)
                        {
                            const float value = static_cast<float>(static_cast<int>((column * 3 + token * 7) % 29) - 14) * 0.0625f;
                            input.row(token)[column] = value;
                            const uint16_t bf = float_to_bfloat16(value);
                            std::memcpy(bf_input.mutable_row_bytes(token).data() + column * sizeof(bf), &bf, sizeof(bf));
                        }
                    for (bool packed : {false, true})
                    {
                        std::shared_ptr<const QnKPack> sidecar;
                        std::array<std::shared_ptr<const QnKPack>*, 1> sidecars{&sidecar};
                        ActivationBuffer output, bf_float_output;
                        ActivationBuffer dispatched;
                        CompiledOperator owner;
                        forward_linear(matrix, input, dispatched, packed ? uint64_t{OptimizationCpuPackedWeights} : uint64_t{0}, &owner, ExecutionBackend::Cpu);
                        ActivationBuffer bf_output(0, 0, DType::BFloat16);
                        block_check(forward_linear_qnk({&matrix}, input, {&output}, packed, sidecars), "native Float32 linear failed");
                        block_check(forward_linear_qnk({&matrix}, bf_input, {&bf_float_output}, packed, sidecars), "native BF16 activation failed");
                        block_check(forward_linear_qnk({&matrix}, bf_input, {&bf_output}, packed, sidecars), "native BF16 output failed");
                        block_check(output.rows() == tokens && output.columns() == rows, "native linear output shape mismatch");
                        for (size_t token = 0; token < tokens; ++token)
                            for (uint32_t row = 0; row < rows; ++row)
                            {
                                const double expected = reference_dot(matrix, row, input.row(token));
                                block_near(output.row(token)[row], expected);
                                block_near(dispatched.row(token)[row], expected);
                                block_near(bf_float_output.row(token)[row], expected);
                                uint16_t bf;
                                std::memcpy(&bf, bf_output.row_bytes(token).data() + row * sizeof(bf), sizeof(bf));
                                block_near(bfloat16_to_float(bf), expected, 0.004);
                            }
                        if (tokens != 0 && packed)
                            block_check(sidecar && sidecar->dtype == dtype, "native compact packed sidecar missing");
                    }
                }
                std::vector<float> embedding(columns + 1, 45678.0f);
                std::vector<float> remapped(columns);
                for (uint32_t row = 0; row < rows; ++row)
                {
                    forward_embedding_row(matrix, row, embedding.data());
                    block_check(embedding.back() == 45678.0f, "embedding decoder wrote beyond one row");
                    TensorData reordered = matrix;
                    reordered.input_column_indices.resize(columns);
                    for (uint32_t column = 0; column < columns; ++column)
                        reordered.input_column_indices[column] = columns - column - 1;
                    forward_embedding_row(reordered, row, remapped.data());
                    const uint8_t* encoded = original.data() + static_cast<size_t>(row) * (columns / block_size) * bytes;
                    for (uint32_t column = 0; column < columns; ++column)
                    {
                        const float expected = reference_value(dtype, encoded + (column / block_size) * bytes, column % block_size);
                        block_check(embedding[column] == expected && remapped[columns - column - 1] == expected, "native embedding or column metadata mismatch");
                    }
                }
                ActivationBuffer map_input(1, columns);
                for (uint32_t column = 0; column < columns; ++column)
                    map_input.row(0)[column] = static_cast<float>(static_cast<int>(column % 31) - 15) * 0.125f;
                TensorData mapped = matrix;
                mapped.input_column_indices.resize(columns);
                for (uint32_t column = 0; column < columns; ++column)
                    mapped.input_column_indices[column] = columns - column - 1;
                ActivationBuffer mapped_output;
                forward_linear(mapped, map_input, mapped_output, 0, nullptr, ExecutionBackend::Cpu);
                const uint8_t* encoded = original.data();
                for (uint32_t row = 0; row < rows; ++row)
                {
                    double expected = 0.0;
                    const uint8_t* row_data = encoded + static_cast<size_t>(row) * (columns / block_size) * bytes;
                    for (uint32_t column = 0; column < columns; ++column)
                        expected += static_cast<double>(reference_value(dtype, row_data + (column / block_size) * bytes, column % block_size))
                                    * map_input.row(0)[columns - column - 1];
                    block_near(mapped_output.row(0)[row], expected);
                }
                block_check(matrix.float32_data.empty() && matrix.bfloat16_data.empty() && matrix.quantized_data.empty()
                                && std::memcmp(matrix.qnk_values().data(), original.data(), original.size()) == 0,
                            "execution expanded or modified immutable mapped native weights");
            }
    }
}

static void test_native_pairs_and_invalid_layouts()
{
    for (DType dtype : {DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::IQ4NL, DType::Q5_1, DType::IQ4XS})
    {
        const uint32_t columns = reference_block_size(dtype) == 32 ? 288 : 512;
        const uint32_t block_size = reference_block_size(dtype);
        const TensorData first = make_blocks(dtype, 9, columns, 1);
        const TensorData second = make_blocks(dtype, 3, columns, 2);
        ActivationBuffer input(5, columns);
        ActivationBuffer bf_input(5, columns, DType::BFloat16);
        for (size_t token = 0; token < input.rows(); ++token)
            for (uint32_t column = 0; column < input.columns(); ++column)
            {
                const float value = static_cast<float>(static_cast<int>((column + token * 5) % 23) - 11) * 0.125f;
                input.row(token)[column] = value;
                const uint16_t bf = float_to_bfloat16(value);
                std::memcpy(bf_input.mutable_row_bytes(token).data() + column * sizeof(bf), &bf, sizeof(bf));
            }
        for (bool packed : {false, true})
        {
            ActivationBuffer left, right, bf_right;
            ActivationBuffer dispatched_left, dispatched_right;
            ActivationBuffer bf_left(0, 0, DType::BFloat16);
            block_check(forward_linear_qnk({&first, &second}, input, {&left, &right}, packed), "native paired Float32 projection failed");
            block_check(forward_linear({&first, &second}, input, {&dispatched_left, &dispatched_right}, packed ? uint64_t{OptimizationCpuPackedWeights} : uint64_t{0}),
                        "native paired CPU dispatch failed");
            block_check(forward_linear_qnk({&first, &second}, bf_input, {&bf_left, &bf_right}, packed), "native paired BF16 projection failed");
            for (size_t token = 0; token < input.rows(); ++token)
            {
                for (uint32_t row = 0; row < first.shape[0]; ++row)
                {
                    const double expected = reference_dot(first, row, input.row(token));
                    block_near(left.row(token)[row], expected);
                    block_near(dispatched_left.row(token)[row], expected);
                    uint16_t bf;
                    std::memcpy(&bf, bf_left.row_bytes(token).data() + row * sizeof(bf), sizeof(bf));
                    block_near(bfloat16_to_float(bf), expected, 0.004);
                }
                for (uint32_t row = 0; row < second.shape[0]; ++row)
                {
                    const double expected = reference_dot(second, row, input.row(token));
                    block_near(right.row(token)[row], expected);
                    block_near(dispatched_right.row(token)[row], expected);
                    block_near(bf_right.row(token)[row], expected);
                }
            }
        }
        ActivationBuffer output;
        TensorData truncated = first;
        truncated.quantized_data.pop_back();
        block_check(!forward_linear_qnk({&truncated}, input, {&output}, false), "truncated native storage accepted");
        block_check(!forward_linear_qnk({&first}, input, {&input}, false), "native input/output alias accepted");
        block_check(!forward_linear_qnk({&first, &second}, input, {&output, &output}, true), "native paired output alias accepted");
        for (uint32_t invalid_columns : {0u, block_size - 1, block_size + 1, 2 * block_size - 1})
            block_check(!qnk_shape_supported(dtype, 1, invalid_columns) && qnk_storage_bytes(dtype, 1, invalid_columns) == 0, "partial native block row accepted");
        const uint32_t wide = std::numeric_limits<uint32_t>::max() - (block_size - 1);
        block_check(qnk_storage_bytes(dtype, std::numeric_limits<size_t>::max(), wide) == 0
                        && qnk_packed_storage_bytes(dtype, std::numeric_limits<size_t>::max(), wide) == 0,
                    "native byte-size overflow accepted");
        QnKPack invalid;
        block_check(!qnk_pack_weights(first.qnk_values().data(), first.qnk_values().size() - 1, dtype, 9, columns, invalid), "truncated native pack accepted");
    }
    block_check(qnk_block_size(DType::Q4_0) == 32 && qnk_block_bytes(DType::Q4_0) == 18
                    && qnk_block_size(DType::Q4_1) == 32 && qnk_block_bytes(DType::Q4_1) == 20
                    && qnk_block_size(DType::Q5_0) == 32 && qnk_block_bytes(DType::Q5_0) == 22
                    && qnk_block_size(DType::IQ4XS) == 256 && qnk_block_bytes(DType::IQ4XS) == 136
                    && qnk_block_size(DType::IQ4NL) == 32 && qnk_block_bytes(DType::IQ4NL) == 18
                    && qnk_block_size(DType::Q5_1) == 32 && qnk_block_bytes(DType::Q5_1) == 24,
                "native GGUF block ABI mismatch");
    // The real PLE table is huge. Validate its compact size without allocating
    // the table; the 160-column fixtures above validate actual single-row reads.
    block_check(qnk_storage_bytes(DType::IQ4NL, 320001536, 160) == UINT64_C(28800138240), "PLE compact row storage size mismatch");
    block_check(qnk_storage_bytes(DType::IQ4XS, 320001536, 256) == UINT64_C(43520208896), "IQ4_XS compact row storage size mismatch");
    for (DType dtype : {DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8K})
        block_check(qnk_block_size(dtype) == 256 && !qnk_shape_supported(dtype, 1, 32), "K-series ABI changed");
    block_check(qnk_block_size(DType::Q8_0) == 32 && qnk_block_bytes(DType::Q8_0) == 34, "Q8_0 ABI changed");
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        ncnn::moe::test_native_block_decoders();
        ncnn::moe::test_native_linear_and_embedding();
        ncnn::moe::test_native_pairs_and_invalid_layouts();
        std::cout << "native GGUF block CPU tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
