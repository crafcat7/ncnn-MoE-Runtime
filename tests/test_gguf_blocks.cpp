#include "graph/compiledoperator.h"
#include "kernels/mxfp4.h"
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

static double reference_ue4m3(uint8_t value)
{
    if (value == 0 || value == 0x7fu)
        return 0.0;
    const int exponent = (value >> 3) & 15;
    const int mantissa = value & 7;
    const double raw = exponent == 0
                           ? std::ldexp(static_cast<double>(mantissa), -9)
                           : std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0, exponent - 7);
    return raw * 0.5;
}

static float reference_nvfp4_value(const uint8_t* source, uint32_t lane)
{
    static constexpr std::array<double, 16> values = {
        0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
        -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0};
    const uint32_t subblock = lane / 16;
    const uint32_t within = lane % 16;
    const uint8_t packed = source[4 + subblock * 8 + within % 8];
    const uint32_t code = (packed >> (within < 8 ? 0 : 4)) & 15u;
    return static_cast<float>(values[code] * reference_ue4m3(source[subblock]));
}

static float reference_mxfp4_gguf_value(uint8_t scale, uint8_t code)
{
    static constexpr std::array<double, 16> values = {
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
    return static_cast<float>(std::ldexp(1.0, static_cast<int>(scale) - 127) * values[code]);
}

static void write_half(uint8_t* output, uint16_t bits)
{
    output[0] = static_cast<uint8_t>(bits);
    output[1] = static_cast<uint8_t>(bits >> 8);
}

static uint32_t reference_block_size(DType dtype)
{
    return dtype == DType::IQ4XS ? 256 : dtype == DType::NvFp4 ? 64
                                                               : 32;
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
    case DType::NvFp4: return 36;
    default: return 0;
    }
}

static float reference_value(DType dtype, const uint8_t* source, uint32_t lane)
{
    // The oracle uses the published GGML byte layout, independent of runtime
    // half conversion, block widths, unpack functions and packed sidecars.
    static constexpr std::array<int, 16> nonlinear = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    if (dtype == DType::NvFp4)
        return reference_nvfp4_value(source, lane);
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
            if (dtype == DType::NvFp4)
            {
                static constexpr std::array<uint8_t, 12> nvfp4_scales = {
                    0x00, 0x01, 0x07, 0x08, 0x38, 0x3f,
                    0x40, 0x7e, 0x7f, 0x80, 0xff, 0x21};
                for (uint32_t subblock = 0; subblock < 4; ++subblock)
                    encoded[subblock] = nvfp4_scales[(row * 3 + block * 5 + subblock + seed) % nvfp4_scales.size()];
                for (uint32_t byte = 0; byte < 32; ++byte)
                    encoded[4 + byte] = static_cast<uint8_t>((byte % 8) | (((byte % 8 + row + block + seed) & 15u) << 4));
                continue;
            }
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

static void test_nvfp4_block_codec()
{
    static constexpr std::array<uint8_t, 32> golden_codes = {
        0x80, 0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7,
        0x80, 0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7,
        0x80, 0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7,
        0x80, 0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7};
    std::array<uint8_t, 36> block{};
    std::copy(golden_codes.begin(), golden_codes.end(), block.begin() + 4);

    std::array<float, 65> decoded;
    std::array<float, 64> input;
    for (uint32_t lane = 0; lane < input.size(); ++lane)
        input[lane] = static_cast<float>(static_cast<int>((lane * 7) % 19) - 9) * 0.125f;

    for (uint16_t encoded_scale = 0; encoded_scale < 256; ++encoded_scale)
    {
        const uint8_t scale = static_cast<uint8_t>(encoded_scale);
        block.fill(0);
        for (uint32_t subblock = 0; subblock < 4; ++subblock)
            block[subblock] = scale;
        std::copy(golden_codes.begin(), golden_codes.end(), block.begin() + 4);
        decoded.fill(98765.0f);
        qnk_dequantize_block(DType::NvFp4, block.data(), decoded.data());
        block_check(decoded[64] == 98765.0f, "NVFP4 decoder wrote beyond its 64-value block");
        double expected_dot = 0.0;
        for (uint32_t lane = 0; lane < 64; ++lane)
        {
            const float expected = reference_nvfp4_value(block.data(), lane);
            block_near(decoded[lane], expected, 1e-6);
            expected_dot += static_cast<double>(expected) * input[lane];
        }
        if (scale != 0x00 && scale != 0x7f && scale != 0x80)
            block_check(std::signbit(decoded[8]), "NVFP4 negative zero sign bit was lost");
        block_near(qnk_dot_block(DType::NvFp4, block.data(), input.data()), expected_dot, 5e-5);
    }

    static constexpr std::array<uint8_t, 4> mixed_scales = {0x01, 0x38, 0x7e, 0xff};
    for (uint32_t subblock = 0; subblock < mixed_scales.size(); ++subblock)
        block[subblock] = mixed_scales[subblock];
    qnk_dequantize_block(DType::NvFp4, block.data(), decoded.data());
    for (uint32_t lane = 0; lane < 64; ++lane)
        block_near(decoded[lane], reference_nvfp4_value(block.data(), lane), 1e-6);

    block_check(is_qnk_dtype(DType::NvFp4) && !is_qnk_dtype(DType::MxFp4), "MXFP4/NVFP4 native QnK classification changed");
    block_check(qnk_block_size(DType::NvFp4) == 64 && qnk_block_bytes(DType::NvFp4) == 36
                    && qnk_shape_supported(DType::NvFp4, 1, 64)
                    && !qnk_shape_supported(DType::NvFp4, 1, 63)
                    && qnk_storage_bytes(DType::NvFp4, 9, 128) == 9u * 2u * 36u,
                "NVFP4 native block ABI mismatch");
    block_check(qnk_block_size(DType::MxFp4) == 32 && qnk_block_bytes(DType::MxFp4) == 17
                    && !qnk_shape_supported(DType::MxFp4, 1, 32)
                    && qnk_storage_bytes(DType::MxFp4, 1, 32) == 0,
                "MXFP4 GPU metadata changed CPU QnK admission");
    std::array<float, 32> mxfp_sentinel;
    mxfp_sentinel.fill(45678.0f);
    qnk_dequantize_block(DType::MxFp4, block.data(), mxfp_sentinel.data());
    block_check(std::all_of(mxfp_sentinel.begin(), mxfp_sentinel.end(), [](float value) { return value == 45678.0f; })
                    && qnk_dot_block(DType::MxFp4, block.data(), input.data()) == 0.0f,
                "MXFP4 metadata entered the native QnK codec");
}

static void test_mxfp4_gguf_codec_and_embedding()
{
    const std::array<uint8_t, 32> canonical = {
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
        0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87,
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78};
    const std::array<uint8_t, 2> scales = {0x12, 0xfe};
    std::array<uint8_t, 34> gguf{};
    block_check(pack_gguf_mxfp4(canonical, scales, gguf), "canonical MXFP4 pack failed");
    static constexpr std::array<uint8_t, 17> expected_first_block = {
        0x12, 0x00, 0xf1, 0x12, 0xe3, 0x24, 0xd5, 0x36, 0xc7,
        0x48, 0xb9, 0x5a, 0xab, 0x6c, 0x9d, 0x7e, 0x8f};
    block_check(std::equal(expected_first_block.begin(), expected_first_block.end(), gguf.begin()),
                "canonical MXFP4 pack bytes mismatch");
    std::array<uint8_t, 32> unpacked_blocks{};
    std::array<uint8_t, 2> unpacked_scales{};
    block_check(unpack_gguf_mxfp4(gguf, unpacked_blocks, unpacked_scales)
                    && unpacked_blocks == canonical && unpacked_scales == scales,
                "canonical MXFP4 pack/unpack roundtrip mismatch");

    std::array<uint8_t, 33> truncated{};
    truncated.fill(0xa5);
    block_check(!pack_gguf_mxfp4(canonical, scales, truncated)
                    && std::all_of(truncated.begin(), truncated.end(), [](uint8_t value) { return value == 0xa5; }),
                "truncated MXFP4 destination accepted or modified");
    std::array<uint8_t, 17> aliased{};
    std::array<uint8_t, 1> separate_scale = {0x7f};
    block_check(!pack_gguf_mxfp4(std::span<const uint8_t>(aliased.data(), 16), separate_scale, aliased),
                "overlapping MXFP4 pack spans accepted");

    TensorData embedding;
    embedding.dtype = DType::MxFp4;
    embedding.shape = {2, 64};
    std::array<uint8_t, 4 * 17> embedding_gguf{};
    for (uint32_t block = 0; block < 4; ++block)
    {
        embedding_gguf[block * 17] = static_cast<uint8_t>(124 + block * 2);
        for (uint32_t byte = 0; byte < 16; ++byte)
            embedding_gguf[block * 17 + 1 + byte] = static_cast<uint8_t>((byte + block) | (((15 - byte + block) & 15u) << 4));
    }
    std::array<uint8_t, 4 * 16> embedding_blocks{};
    std::array<uint8_t, 4> embedding_scales{};
    block_check(unpack_gguf_mxfp4(embedding_gguf, embedding_blocks, embedding_scales), "MXFP4 embedding fixture unpack failed");
    embedding.mxfp4_blocks.assign(embedding_blocks.data(), embedding_blocks.size());
    embedding.mxfp4_scales.assign(embedding_scales.data(), embedding_scales.size());

    std::array<float, 65> output;
    output.fill(45678.0f);
    for (uint32_t row = 0; row < 2; ++row)
    {
        forward_embedding_row(embedding, row, output.data());
        block_check(output.back() == 45678.0f, "MXFP4 embedding decoder wrote beyond one row");
        for (uint32_t column = 0; column < 64; ++column)
        {
            const uint8_t* source = embedding_gguf.data() + (row * 2 + column / 32) * 17;
            const uint8_t packed = source[1 + column % 16];
            const uint32_t code = (packed >> ((column % 32) / 16 * 4)) & 15u;
            block_near(output[column], reference_mxfp4_gguf_value(source[0], code), 1e-6);
        }
    }
    TensorData remapped = embedding;
    remapped.input_column_indices.resize(64);
    for (uint32_t column = 0; column < 64; ++column)
        remapped.input_column_indices[column] = 63 - column;
    forward_embedding_row(remapped, 1, output.data());
    for (uint32_t column = 0; column < 64; ++column)
    {
        const uint8_t* source = embedding_gguf.data() + (2 + column / 32) * 17;
        const uint8_t packed = source[1 + column % 16];
        const uint32_t code = (packed >> ((column % 32) / 16 * 4)) & 15u;
        block_near(output[63 - column], reference_mxfp4_gguf_value(source[0], code), 1e-6);
    }

    ActivationBuffer input(1, 64);
    for (uint32_t column = 0; column < input.columns(); ++column)
        input.row(0)[column] = static_cast<float>(static_cast<int>(column % 17) - 8) * 0.125f;
    ActivationBuffer projected;
    const uint64_t scalar_flags = OptimizationDefaultFlags & ~static_cast<uint64_t>(OptimizationCpuMxfp4Q8);
    forward_linear(remapped, input, projected, scalar_flags, nullptr, ExecutionBackend::Cpu);
    for (uint32_t row = 0; row < 2; ++row)
    {
        double expected = 0.0;
        for (uint32_t column = 0; column < 64; ++column)
        {
            const uint8_t* source = embedding_gguf.data() + (row * 2 + column / 32) * 17;
            const uint8_t packed = source[1 + column % 16];
            const uint32_t code = (packed >> ((column % 32) / 16 * 4)) & 15u;
            expected += reference_mxfp4_gguf_value(source[0], code) * input.row(0)[63 - column];
        }
        block_near(projected.row(0)[row], expected, 1e-5);
    }
    TensorData invalid_mapping = remapped;
    invalid_mapping.input_column_indices[0] = 64;
    bool rejected = false;
    try
    {
        ActivationBuffer invalid_output;
        forward_linear(invalid_mapping, input, invalid_output, scalar_flags, nullptr, ExecutionBackend::Cpu);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    block_check(rejected, "MXFP4 out-of-range input column map accepted");

    invalid_mapping = remapped;
    invalid_mapping.input_column_indices[0] = invalid_mapping.input_column_indices[1];
    for (bool embedding_row : {false, true})
    {
        rejected = false;
        try
        {
            if (embedding_row)
                forward_embedding_row(invalid_mapping, 0, output.data());
            else
            {
                ActivationBuffer invalid_output;
                forward_linear(invalid_mapping, input, invalid_output, scalar_flags, nullptr, ExecutionBackend::Cpu);
            }
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        block_check(rejected, "MXFP4 duplicate input column map accepted");
    }

    TensorData truncated_embedding = embedding;
    truncated_embedding.mxfp4_scales.resize(embedding.mxfp4_scales.size() - 1);
    rejected = false;
    try
    {
        forward_embedding_row(truncated_embedding, 0, output.data());
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    block_check(rejected, "truncated MXFP4 embedding storage accepted");

    TensorData extreme_scale_embedding;
    extreme_scale_embedding.dtype = DType::MxFp4;
    extreme_scale_embedding.shape = {1, 32};
    std::array<uint8_t, 16> extreme_codes{};
    extreme_codes[0] = 0x01;
    extreme_scale_embedding.mxfp4_blocks.assign(extreme_codes.data(), extreme_codes.size());
    extreme_scale_embedding.mxfp4_scales.resize(1);
    for (uint8_t scale : {uint8_t{0}, uint8_t{1}, uint8_t{255}})
    {
        extreme_scale_embedding.mxfp4_scales[0] = scale;
        forward_embedding_row(extreme_scale_embedding, 0, output.data());
        const int expected_exponent = scale == 0 ? -128 : scale == 1 ? -127
                                                                     : 127;
        block_check(output[0] == std::ldexp(1.0f, expected_exponent) && output[1] == 0.0f
                        && std::all_of(output.begin(), output.begin() + 32, [](float value) { return !std::isnan(value); }),
                    "MXFP4 E8M0 embedding scale edge mismatch");
    }
}

static void test_mxfp4_scale255_cpu_paths()
{
    std::array<uint8_t, 16> zero_codes{};
    std::array<uint8_t, 16> code_one{};
    code_one[0] = 0x01;
    const std::array<uint8_t, 1> scale255 = {0xff};
    std::array<float, 32> float_input;
    float_input.fill(1.0f);
    float_input[0] = 0.25f;
    const float expected_weight = std::ldexp(1.0f, 127);
    const float expected_dot = std::ldexp(1.0f, 125);
    block_check(std::isfinite(expected_weight) && std::isfinite(expected_dot), "MXFP4 scale-255 reference is not finite");

    block_check(mxfp4_dot(zero_codes.data(), scale255.data(), 1, float_input.data()) == 0.0f,
                "MXFP4 zero code with scale 255 produced a non-finite dot");
    block_check(mxfp4_dot(code_one.data(), scale255.data(), 1, float_input.data()) == expected_dot,
                "MXFP4 scale-255 dot mismatch");

    std::array<float, 64> float_tokens{};
    float_tokens[0] = 0.25f;
    float_tokens[32] = 1.0f;
    std::array<float, 4> gemm_output = {45678.0f, 45678.0f, 45678.0f, 45678.0f};
    mxfp4_gemm_row(code_one.data(), scale255.data(), 1, float_tokens.data(), 32, 2, gemm_output.data(), 2);
    block_check(gemm_output[0] == expected_dot && gemm_output[2] == expected_weight
                    && gemm_output[1] == 45678.0f && gemm_output[3] == 45678.0f,
                "MXFP4 scale-255 GEMM mismatch");
    mxfp4_gemm_row(zero_codes.data(), scale255.data(), 1, float_tokens.data(), 32, 2, gemm_output.data(), 2);
    block_check(gemm_output[0] == 0.0f && gemm_output[2] == 0.0f,
                "MXFP4 zero code with scale 255 produced a non-finite GEMM value");

    std::array<int8_t, 32> q8_input{};
    q8_input[0] = 1;
    const std::array<float, 1> q8_input_scale = {0.25f};
    block_check(mxfp4_q8_dot(zero_codes.data(), scale255.data(), 1, q8_input.data(), q8_input_scale.data()) == 0.0f,
                "MXFP4 zero code with scale 255 produced a non-finite Q8 dot");
    block_check(mxfp4_q8_dot(code_one.data(), scale255.data(), 1, q8_input.data(), q8_input_scale.data()) == expected_dot,
                "MXFP4 scale-255 Q8 dot mismatch");

    std::array<int8_t, 64> q8_tokens{};
    q8_tokens[0] = 1;
    const std::array<float, 2> q8_token_scales = {0.25f, 0.25f};
    std::array<float, 2> q8_gemm_output = {45678.0f, 45678.0f};
    mxfp4_q8_gemm_row(code_one.data(), scale255.data(), 1, q8_tokens.data(), 32,
                      q8_token_scales.data(), 1, 2, q8_gemm_output.data(), 1);
    block_check(q8_gemm_output[0] == expected_dot && q8_gemm_output[1] == 0.0f,
                "MXFP4 scale-255 Q8 GEMM mismatch");
    mxfp4_q8_gemm_row(zero_codes.data(), scale255.data(), 1, q8_tokens.data(), 32,
                      q8_token_scales.data(), 1, 2, q8_gemm_output.data(), 1);
    block_check(q8_gemm_output[0] == 0.0f && q8_gemm_output[1] == 0.0f,
                "MXFP4 zero code with scale 255 produced a non-finite Q8 GEMM value");

    std::array<uint8_t, 32> rows_packed{};
    std::copy(code_one.begin(), code_one.end(), rows_packed.begin() + 16);
    const std::array<uint8_t, 2> row_scales = {0xff, 0xff};
    Mxfp4Q8PackedMatrix packed_weights;
    block_check(mxfp4_q8_pack_weights(rows_packed.data(), row_scales.data(), 1, 2, packed_weights, 4),
                "MXFP4 scale-255 Q8 weight pack failed");
    block_check(packed_weights.storage[0] == 0xff && packed_weights.storage[1] == 0xff,
                "MXFP4 Q8 weight pack changed the scale-255 bytes");
    std::array<float, 4> packed_gemv_output = {45678.0f, 45678.0f, 45678.0f, 45678.0f};
    mxfp4_q8_packed_gemv(packed_weights, q8_input.data(), q8_input_scale.data(), packed_gemv_output.data());
    block_check(packed_gemv_output[0] == 0.0f && packed_gemv_output[1] == expected_dot
                    && packed_gemv_output[2] == 45678.0f && packed_gemv_output[3] == 45678.0f,
                "MXFP4 scale-255 packed Q8 GEMV mismatch");
    std::array<float, 2> packed_token_scales = {0.25f, 0.25f};
    std::array<int8_t, 64> packed_tokens{};
    packed_tokens[0] = 1;
    std::array<float, 8> packed_gemm_output;
    packed_gemm_output.fill(45678.0f);
    mxfp4_q8_packed_gemm(packed_weights, packed_tokens.data(), 64, packed_token_scales.data(), 1,
                         2, packed_gemm_output.data(), 4);
    block_check(packed_gemm_output[0] == 0.0f && packed_gemm_output[1] == expected_dot
                    && packed_gemm_output[2] == 45678.0f && packed_gemm_output[3] == 45678.0f
                    && packed_gemm_output[4] == 0.0f && packed_gemm_output[5] == 0.0f
                    && packed_gemm_output[6] == 45678.0f && packed_gemm_output[7] == 45678.0f,
                "MXFP4 scale-255 packed Q8 GEMM mismatch");

    std::array<uint8_t, 16> canonical{};
    canonical[0] = 0x01;
    const std::array<uint8_t, 1> canonical_scale = {0xff};
    std::array<uint8_t, 17> gguf{};
    block_check(pack_gguf_mxfp4(canonical, canonical_scale, gguf)
                    && gguf[0] == 0xff && gguf[1] == 0x01 && gguf[2] == 0,
                "MXFP4 scale-255 canonical pack bytes mismatch");
    block_check(std::isfinite(reference_mxfp4_gguf_value(gguf[0], 1))
                    && reference_mxfp4_gguf_value(gguf[0], 1) == expected_weight
                    && reference_mxfp4_gguf_value(gguf[0], 0) == 0.0f,
                "MXFP4 scale-255 packed zero/code-one references mismatch");
    std::array<float, 32> decoded{};
    mxfp4_dequantize_block(canonical.data(), 0xff, decoded.data());
    block_check(decoded[0] == expected_weight && decoded[1] == 0.0f
                    && std::all_of(decoded.begin(), decoded.end(), [](float value) { return std::isfinite(value); }),
                "MXFP4 scale-255 canonical decoder mismatch");
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
    for (DType dtype : {DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::IQ4NL, DType::Q5_1, DType::IQ4XS, DType::NvFp4})
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
                for (uint32_t row = rows; row < (rows + 7) / 8 * 8; ++row)
                    for (uint32_t block = 0; block < columns / block_size; ++block)
                    {
                        const uint8_t* padding = qnk_packed_block(pack, row, block);
                        block_check(std::all_of(padding, padding + bytes, [](uint8_t value) { return value == 0; }),
                                    "native packed row-tail padding was not zeroed");
                    }
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
    for (DType dtype : {DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::IQ4NL, DType::Q5_1, DType::IQ4XS, DType::NvFp4})
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
        ncnn::moe::test_nvfp4_block_codec();
        ncnn::moe::test_mxfp4_gguf_codec_and_embedding();
        ncnn::moe::test_mxfp4_scale255_cpu_paths();
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
