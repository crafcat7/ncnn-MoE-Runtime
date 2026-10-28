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
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static void block_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static TensorData block_matrix(DType dtype, uint32_t rows, uint32_t columns, uint32_t seed, bool extreme = false)
{
    TensorData matrix;
    matrix.dtype = dtype;
    matrix.shape = {rows, columns};
    if (dtype == DType::MxFp4)
    {
        const size_t blocks = static_cast<size_t>(rows) * (columns / 32);
        matrix.mxfp4_blocks.resize(blocks * 16);
        matrix.mxfp4_scales.resize(blocks);
        for (size_t block = 0; block < blocks; ++block)
        {
            matrix.mxfp4_scales[block] = static_cast<uint8_t>(116u + (block * 3 + seed) % 8u);
            for (uint32_t pair = 0; pair < 8; ++pair)
            {
                const uint8_t low = static_cast<uint8_t>((pair * 3 + block + seed) & 15u);
                const uint8_t high = static_cast<uint8_t>((pair * 7 + block * 5 + seed + 1) & 15u);
                matrix.mxfp4_blocks[block * 16 + pair] = static_cast<uint8_t>(low | (high << 4));
                const uint8_t upper_low = static_cast<uint8_t>((pair * 11 + block * 3 + seed + 2) & 15u);
                const uint8_t upper_high = static_cast<uint8_t>((pair * 13 + block * 7 + seed + 3) & 15u);
                matrix.mxfp4_blocks[block * 16 + pair + 8] = static_cast<uint8_t>(upper_low | (upper_high << 4));
            }
        }
        return matrix;
    }
    if (dtype == DType::NvFp4)
    {
        const size_t bytes = static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns));
        block_check(bytes == static_cast<size_t>(rows) * (columns / 64) * 36, "NVFP4 native byte count");
        matrix.quantized_data.resize(bytes);
        for (size_t block = 0; block < bytes / 36; ++block)
        {
            uint8_t* encoded = matrix.quantized_data.data() + block * 36;
            for (uint32_t subblock = 0; subblock < 4; ++subblock)
                encoded[subblock] = static_cast<uint8_t>((block * 4 + subblock + seed * 17) & 255u);
            for (uint32_t byte = 0; byte < 32; ++byte)
            {
                const uint8_t low = static_cast<uint8_t>((byte * 3 + block + seed) & 15u);
                const uint8_t high = static_cast<uint8_t>((byte * 11 + block * 7 + seed + 1) & 15u);
                encoded[4 + byte] = static_cast<uint8_t>(low | (high << 4));
            }
        }
        return matrix;
    }
    const size_t block_bytes = dtype == DType::IQ4NL || dtype == DType::Q4_0 ? 18
                               : dtype == DType::Q4_1                        ? 20
                               : dtype == DType::Q5_1                        ? 24
                               : dtype == DType::Q5_0                        ? 22
                               : dtype == DType::IQ4XS                       ? 136
                               : dtype == DType::Q4K                         ? 144
                                                                             : 34;
    const size_t size = static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns));
    block_check(size == static_cast<size_t>(rows) * (columns / qnk_block_size(dtype)) * block_bytes, "native GGUF block byte count");
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
        else if (dtype == DType::Q4_0 || dtype == DType::Q4_1)
        {
            if (dtype == DType::Q4_1)
            {
                const uint16_t minimum = block % 3 == 0 ? 0x9400 : block % 3 == 1 ? 0x1400
                                                                                  : 0x8001;
                encoded[2] = static_cast<uint8_t>(minimum);
                encoded[3] = static_cast<uint8_t>(minimum >> 8);
            }
            const uint32_t quant_offset = dtype == DType::Q4_0 ? 2u : 4u;
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const uint8_t code = static_cast<uint8_t>((lane * 7 + block + seed) & 15);
                encoded[quant_offset + lane % 16] |= static_cast<uint8_t>(code << (lane < 16 ? 0 : 4));
            }
        }
        else if (dtype == DType::Q5_0)
        {
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const uint8_t code = static_cast<uint8_t>((lane * 5 + block + seed) & 31);
                encoded[6 + lane % 16] |= static_cast<uint8_t>((code & 15) << (lane < 16 ? 0 : 4));
                encoded[2 + lane / 8] |= static_cast<uint8_t>((code >> 4) << (lane % 8));
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
        else if (dtype == DType::IQ4XS)
        {
            uint16_t scale_high = 0;
            for (uint32_t subblock = 0; subblock < 8; ++subblock)
            {
                const uint8_t scale_code = static_cast<uint8_t>((subblock * 9 + block + seed) % 63 + 1);
                encoded[4 + subblock / 2] |= static_cast<uint8_t>((scale_code & 15) << ((subblock & 1) * 4));
                scale_high |= static_cast<uint16_t>(scale_code >> 4) << (subblock * 2);
                for (uint32_t lane = 0; lane < 32; ++lane)
                {
                    const uint8_t code = static_cast<uint8_t>((lane * 5 + subblock * 3 + block + seed) & 15);
                    encoded[8 + subblock * 16 + lane % 16] |= static_cast<uint8_t>(code << (lane < 16 ? 0 : 4));
                }
            }
            encoded[2] = static_cast<uint8_t>(scale_high);
            encoded[3] = static_cast<uint8_t>(scale_high >> 8);
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

static ActivationBuffer block_input(size_t rows, uint32_t columns)
{
    ActivationBuffer input(rows, columns);
    for (size_t token = 0; token < rows; ++token)
        for (uint32_t column = 0; column < columns; ++column)
            input.row(token)[column] = static_cast<float>(static_cast<int>((token * 13 + column * 7 + column / 5) % 41) - 20) / 32.f;
    return input;
}

// Independent GGML-format oracle: no runtime dequantizer, half converter or
// qnk dot routine participates in the expected result.
static float half_to_float(uint8_t low, uint8_t high)
{
    const uint32_t bits = static_cast<uint32_t>(low) | (static_cast<uint32_t>(high) << 8);
    const uint32_t exponent = (bits >> 10) & 31;
    const uint32_t fraction = bits & 1023;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24)
                                          : std::ldexp(static_cast<float>(1024 + fraction), static_cast<int>(exponent) - 25);
    return bits & 32768 ? -magnitude : magnitude;
}

static float mxfp4_scale(uint8_t exponent)
{
    const uint32_t bits = exponent < 2 ? (0x00200000u << exponent)
                                       : (static_cast<uint32_t>(exponent - 1) << 23);
    float value = 0.f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static float ue4m3_scale(uint8_t encoded)
{
    if (encoded == 0 || encoded == 0x7f)
        return 0.f;
    const uint32_t exponent = (encoded >> 3) & 15u;
    const uint32_t mantissa = encoded & 7u;
    const float raw = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                                    : std::ldexp(1.f + static_cast<float>(mantissa) / 8.f, static_cast<int>(exponent) - 7);
    return raw * 0.5f;
}

static ActivationBuffer block_reference(const TensorData& matrix, const ActivationBuffer& input, uint32_t groups = 1)
{
    constexpr std::array<int, 16> codebook = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    constexpr std::array<int, 16> doubled = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
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
                const size_t block_bytes = matrix.dtype == DType::IQ4NL || matrix.dtype == DType::Q4_0 ? 18
                                           : matrix.dtype == DType::Q4_1                               ? 20
                                           : matrix.dtype == DType::Q5_1                               ? 24
                                           : matrix.dtype == DType::Q5_0                               ? 22
                                           : matrix.dtype == DType::IQ4XS                              ? 136
                                           : matrix.dtype == DType::Q4K                                ? 144
                                                                                                       : 34;
                const uint32_t block_columns = matrix.dtype == DType::Q4K || matrix.dtype == DType::IQ4XS ? 256 : 32;
                const size_t offset = (static_cast<size_t>(row) * (columns / block_columns) + column / block_columns) * block_bytes;
                const uint32_t lane = column % block_columns;
                const float scale = matrix.dtype == DType::MxFp4 || matrix.dtype == DType::NvFp4
                                        ? 0.f
                                        : half_to_float(raw[offset], raw[offset + 1]);
                float value;
                if (matrix.dtype == DType::MxFp4)
                {
                    const size_t block = static_cast<size_t>(row) * (columns / 32) + column / 32;
                    const uint32_t lane = column % 32;
                    const uint8_t packed = matrix.mxfp4_blocks[block * 16 + lane / 2];
                    const uint32_t code = (packed >> ((lane & 1u) * 4u)) & 15u;
                    value = mxfp4_scale(matrix.mxfp4_scales[block]) * doubled[code];
                }
                else if (matrix.dtype == DType::NvFp4)
                {
                    const size_t block = static_cast<size_t>(row) * (columns / 64) + column / 64;
                    const uint32_t lane = column % 64;
                    const uint32_t subblock = lane / 16;
                    const uint32_t sublane = lane % 16;
                    const uint8_t packed = raw[block * 36 + 4 + subblock * 8 + (sublane & 7u)];
                    const uint32_t code = (packed >> (sublane < 8 ? 0u : 4u)) & 15u;
                    value = ue4m3_scale(raw[block * 36 + subblock]) * doubled[code];
                }
                else if (matrix.dtype == DType::Q4K)
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
                    value = scale * code + half_to_float(raw[offset + 2], raw[offset + 3]);
                }
                else if (matrix.dtype == DType::Q4_0 || matrix.dtype == DType::Q4_1)
                {
                    const size_t quant_offset = matrix.dtype == DType::Q4_0 ? 2 : 4;
                    const uint8_t packed = raw[offset + quant_offset + lane % 16];
                    const uint32_t code = lane < 16 ? packed & 15 : packed >> 4;
                    value = matrix.dtype == DType::Q4_0
                                ? scale * (static_cast<int>(code) - 8)
                                : scale * code + half_to_float(raw[offset + 2], raw[offset + 3]);
                }
                else if (matrix.dtype == DType::Q5_0)
                {
                    const uint8_t packed = raw[offset + 6 + lane % 16];
                    const uint32_t low = lane < 16 ? packed & 15 : packed >> 4;
                    const uint32_t high = (raw[offset + 2 + lane / 8] >> (lane % 8)) & 1;
                    value = scale * (static_cast<int>(low | (high << 4)) - 16);
                }
                else if (matrix.dtype == DType::IQ4XS)
                {
                    const uint32_t subblock = lane / 32;
                    const uint32_t sublane = lane % 32;
                    const uint8_t packed = raw[offset + 8 + subblock * 16 + sublane % 16];
                    const uint32_t code = sublane < 16 ? packed & 15 : packed >> 4;
                    const uint32_t scale_high = (static_cast<uint32_t>(raw[offset + 2])
                                                 | (static_cast<uint32_t>(raw[offset + 3]) << 8))
                                                    >> (subblock * 2)
                                                & 3;
                    const uint32_t scale_low = raw[offset + 4 + subblock / 2] >> ((subblock & 1) * 4) & 15;
                    value = scale * (static_cast<int>((scale_high << 4) | scale_low) - 32) * codebook[code];
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

static void block_compare(const ActivationBuffer& actual, const ActivationBuffer& expected, const std::string& label,
                          float absolute_tolerance = 0.00002f, float relative_tolerance = 0.00003f)
{
    block_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + ": shape");
    for (size_t index = 0; index < expected.values().size(); ++index)
    {
        const float reference = expected.values()[index];
        block_check(std::isfinite(actual.values()[index])
                        && std::abs(actual.values()[index] - reference)
                               <= absolute_tolerance + relative_tolerance * std::abs(reference),
                    label + ": element " + std::to_string(index) + " actual=" + std::to_string(actual.values()[index])
                        + " expected=" + std::to_string(reference));
    }
}

static void block_compare_expert(const ActivationBuffer& actual, const ActivationBuffer& expected,
                                 const std::string& label, DType dtype, DType down_dtype)
{
    const bool fp4 = dtype == DType::MxFp4 || dtype == DType::NvFp4;
    // IQ4_XS local scales amplify SiLU rounding before the down projection.
    // Cancellation can leave small outputs despite large intermediate terms;
    // allow chain roundoff while retaining strict single-projection checks.
    const bool iq4xs = dtype == DType::IQ4XS || down_dtype == DType::IQ4XS;
    block_compare(actual, expected, label + " dtype=" + std::to_string(static_cast<uint32_t>(dtype)) + " down_dtype=" + std::to_string(static_cast<uint32_t>(down_dtype)),
                  fp4 || iq4xs ? 0.001f : 0.00002f, fp4 ? 0.00005f : 0.00003f);
}

static void block_projections(uint32_t device, DType dtype)
{
    const bool fp4 = dtype == DType::MxFp4 || dtype == DType::NvFp4;
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    block_check(static_cast<bool>(context), "native block projection context");
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    const auto align = [alignment](uint64_t value) { return (value + alignment - 1) / alignment * alignment; };
    const uint32_t block_columns = qnk_block_size(dtype);
    for (uint32_t block_count : {1u, 2u, 3u, 9u})
    {
        if (fp4 && block_count == 9)
            continue;
        for (uint32_t groups : {1u, 2u})
        {
            const uint32_t columns = block_columns * block_count;
            TensorData matrix = block_matrix(dtype, groups == 1 ? 5 : 6, columns, columns);
            TensorData bias;
            const TensorData* bias_ptr = nullptr;
            if (fp4)
            {
                matrix.input_column_indices.resize(columns);
                for (uint32_t column = 0; column < columns; ++column)
                    matrix.input_column_indices[column] = columns - column - 1;
                bias.dtype = DType::Float32;
                bias.shape = {matrix.shape[0]};
                bias.float32_data.resize(matrix.shape[0]);
                for (uint32_t row = 0; row < matrix.shape[0]; ++row)
                    bias.float32_data[row] = static_cast<float>(static_cast<int>(row % 5) - 2) / 16.f;
                bias_ptr = &bias;
            }
            const uint64_t native_bytes = dtype == DType::MxFp4
                                              ? matrix.mxfp4_blocks.size() + matrix.mxfp4_scales.size()
                                              : matrix.qnk_values().size();
            const uint64_t expected_storage = align(native_bytes) + align(matrix.shape[0] * sizeof(float))
                                              + (fp4 ? align(matrix.input_column_indices.size() * sizeof(uint32_t)) : 0);
            auto operation = QnkLinear_vulkan::create(matrix, bias_ptr, device, runtime, OptimizationDefaultFlags, groups);
            block_check(operation && operation->dtype() == dtype
                            && operation->device_bytes() == expected_storage
                            && QnkLinear_vulkan::storage_bytes(matrix, alignment) == expected_storage,
                        "native block projection stores compact weights, optional bias and map");
            for (size_t tokens : {size_t(1), size_t(4), size_t(5)})
            {
                if (fp4 && tokens == 5)
                    continue;
                const auto input = block_input(tokens, columns * groups);
                auto expected = block_reference(matrix, input, groups);
                if (fp4)
                    for (size_t token = 0; token < input.rows(); ++token)
                        for (uint32_t row = 0; row < matrix.shape[0]; ++row)
                            expected.row(token)[row] += bias.float32_data[row];
                ActivationBuffer actual;
                const auto before = get_vulkan_statistics(runtime);
                block_check(operation->forward(input, actual), "native block projection dispatch");
                if (!fp4)
                    block_check(get_vulkan_statistics(runtime).dispatches == before.dispatches + 1,
                                "native block projection uses one dispatch");
                block_compare(actual, expected, "native grouped/mapped/bias/block-tail reference",
                              fp4 ? 0.001f : 0.00002f, fp4 ? 0.00005f : 0.00003f);
                if (!fp4 && groups == 1)
                    block_compare(forward_linear(matrix, input, OptimizationDefaultFlags), expected, "IQ4_NL/Q5_1 CPU native reference");
            }
        }
    }
    if (dtype == DType::NvFp4)
    {
        // Four scale bytes per block cover every UE4M3 encoding.
        TensorData matrix = block_matrix(dtype, 64, 64, 0);
        for (size_t block = 0; block < 64; ++block)
            for (uint32_t subblock = 0; subblock < 4; ++subblock)
                matrix.quantized_data[block * 36 + subblock] = static_cast<uint8_t>(block * 4 + subblock);
        auto operation = QnkLinear_vulkan::create(matrix, nullptr, device, runtime, OptimizationDefaultFlags);
        ActivationBuffer input(1, 64);
        std::fill_n(input.row(0), input.columns(), 1.f / 4096.f);
        ActivationBuffer actual;
        block_check(operation && operation->forward(input, actual), "NVFP4 all-scale projection dispatch");
        block_compare(actual, block_reference(matrix, input), "NVFP4 all 256 UE4M3 encodings", 0.001f, 0.00005f);
    }
    TensorData invalid = block_matrix(dtype, 5, block_columns, 1);
    if (dtype == DType::MxFp4)
    {
        invalid.mxfp4_blocks.resize(invalid.mxfp4_blocks.size() - 1);
        block_check(QnkLinear_vulkan::storage_bytes(invalid, alignment) == 0
                        && !QnkLinear_vulkan::create(invalid, nullptr, device, runtime, OptimizationDefaultFlags),
                    "reject truncated MXFP4 bytes");
        invalid = block_matrix(dtype, 5, block_columns, 1);
        invalid.bfloat16_data.resize(invalid.element_count());
        block_check(QnkLinear_vulkan::storage_bytes(invalid, alignment) == 0
                        && !QnkLinear_vulkan::create(invalid, nullptr, device, runtime, OptimizationDefaultFlags),
                    "reject MXFP4 with an unrelated BF16 copy");
    }
    else if (dtype == DType::NvFp4)
    {
        invalid.quantized_data.resize(invalid.quantized_data.size() - 1);
        block_check(QnkLinear_vulkan::storage_bytes(invalid, alignment) == 0
                        && !QnkLinear_vulkan::create(invalid, nullptr, device, runtime, OptimizationDefaultFlags),
                    "reject truncated NVFP4 bytes");
    }
    else
    {
        for (bool extreme : {false, true})
        {
            const uint32_t columns = block_columns * 3;
            TensorData mapped = block_matrix(dtype, 5, columns, 7, extreme);
            mapped.input_column_indices.resize(columns);
            for (uint32_t column = 0; column < columns; ++column)
                mapped.input_column_indices[column] = columns - column - 1;
            auto operation = QnkLinear_vulkan::create(mapped, nullptr, device, runtime, OptimizationDefaultFlags);
            const auto input = block_input(4, columns);
            ActivationBuffer actual;
            block_check(operation && operation->forward(input, actual), "native column-mapped projection");
            block_compare(actual, block_reference(mapped, input), "native column map and finite extreme scales");
        }
        invalid.shape[1] = block_columns + 1;
        block_check(!QnkLinear_vulkan::create(invalid, nullptr, device, runtime, OptimizationDefaultFlags),
                    "reject incomplete native block");
        invalid = block_matrix(dtype, 5, block_columns, 1);
        --invalid.mapped_size;
        block_check(!QnkLinear_vulkan::create(invalid, nullptr, device, runtime, OptimizationDefaultFlags),
                    "reject truncated native bytes");
        constexpr uint32_t vocabulary_rows = 65537;
        const TensorData vocabulary = block_matrix(dtype, vocabulary_rows, block_columns, 1);
        auto head = QnkLinear_vulkan::create(vocabulary, nullptr, device, runtime, OptimizationDefaultFlags);
        const auto input = block_input(1, block_columns);
        ActivationBuffer logits;
        block_check(head && head->forward(input, logits), "native GGUF LM Head projection");
        block_compare(logits, block_reference(vocabulary, input), "native GGUF tiled LM Head reference");
    }
}

static ActivationBuffer block_expert_reference(const TensorData& gate, const TensorData* up, const TensorData& down,
                                               const ActivationBuffer& input, bool interleaved, float limit)
{
    const auto projected = block_reference(gate, input);
    const auto projected_up = up ? block_reference(*up, input) : ActivationBuffer{};
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
    return block_reference(down, activated);
}

static TensorData fp4_bias(uint32_t rows, uint32_t seed)
{
    TensorData bias;
    bias.dtype = DType::Float32;
    bias.shape = {rows};
    bias.float32_data.resize(rows);
    for (uint32_t row = 0; row < rows; ++row)
        bias.float32_data[row] = static_cast<float>(static_cast<int>((row * 3 + seed) % 7) - 3) / 64.f;
    return bias;
}

static TensorData packed_gate_up(const TensorData& gate, const TensorData& up)
{
    block_check(gate.dtype == up.dtype && gate.shape.size() == 2 && up.shape == gate.shape,
                "packed FP4 gate/up source shapes");
    TensorData packed;
    packed.dtype = gate.dtype;
    packed.shape = {gate.shape[0] * 2, gate.shape[1]};
    packed.qnk_interleave_rows = false;
    if (gate.dtype == DType::MxFp4)
    {
        const size_t block_row_bytes = static_cast<size_t>(gate.shape[1] / 32) * 16;
        const size_t scale_row_bytes = gate.shape[1] / 32;
        packed.mxfp4_blocks.resize(static_cast<size_t>(packed.shape[0]) * block_row_bytes);
        packed.mxfp4_scales.resize(static_cast<size_t>(packed.shape[0]) * scale_row_bytes);
        for (uint32_t row = 0; row < gate.shape[0]; ++row)
        {
            std::memcpy(packed.mxfp4_blocks.data() + static_cast<size_t>(row) * block_row_bytes,
                        gate.mxfp4_blocks.data() + static_cast<size_t>(row) * block_row_bytes,
                        block_row_bytes);
            std::memcpy(packed.mxfp4_blocks.data() + static_cast<size_t>(row + gate.shape[0]) * block_row_bytes,
                        up.mxfp4_blocks.data() + static_cast<size_t>(row) * block_row_bytes,
                        block_row_bytes);
            std::memcpy(packed.mxfp4_scales.data() + static_cast<size_t>(row) * scale_row_bytes,
                        gate.mxfp4_scales.data() + static_cast<size_t>(row) * scale_row_bytes,
                        scale_row_bytes);
            std::memcpy(packed.mxfp4_scales.data() + static_cast<size_t>(row + gate.shape[0]) * scale_row_bytes,
                        up.mxfp4_scales.data() + static_cast<size_t>(row) * scale_row_bytes,
                        scale_row_bytes);
        }
    }
    else
    {
        const size_t row_bytes = static_cast<size_t>(qnk_block_bytes(gate.dtype))
                                 * (gate.shape[1] / qnk_block_size(gate.dtype));
        const auto gate_bytes = gate.qnk_values();
        const auto up_bytes = up.qnk_values();
        block_check(gate_bytes.size() == static_cast<size_t>(gate.shape[0]) * row_bytes
                        && up_bytes.size() == gate_bytes.size(),
                    "packed NVFP4 source byte counts");
        packed.quantized_data.resize(static_cast<size_t>(packed.shape[0]) * row_bytes);
        for (uint32_t row = 0; row < gate.shape[0]; ++row)
        {
            std::memcpy(packed.quantized_data.data() + static_cast<size_t>(row) * row_bytes,
                        gate_bytes.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);
            std::memcpy(packed.quantized_data.data() + static_cast<size_t>(row + gate.shape[0]) * row_bytes,
                        up_bytes.data() + static_cast<size_t>(row) * row_bytes,
                        row_bytes);
        }
    }
    return packed;
}

static std::shared_ptr<DeviceTensor_vulkan> block_upload(const ActivationBuffer& input,
                                                         uint32_t device,
                                                         const VulkanRuntimePtr& runtime);

static void mixed_fp4_experts(uint32_t device)
{
    constexpr uint32_t hidden = 64;
    constexpr uint32_t intermediate = 64;
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    block_check(static_cast<bool>(context), "mixed MXFP4/NVFP4 Expert context");
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    const std::array<std::array<DType, 2>, 2> mixed_formats = {{
        {DType::MxFp4, DType::NvFp4},
        {DType::NvFp4, DType::MxFp4},
    }};
    for (size_t format = 0; format < mixed_formats.size(); ++format)
    {
        const DType gate_up_dtype = mixed_formats[format][0];
        const DType down_dtype = mixed_formats[format][1];
        const TensorData gate = block_matrix(gate_up_dtype, intermediate, hidden, 11 + format * 3);
        const TensorData up = block_matrix(gate_up_dtype, intermediate, hidden, 12 + format * 3);
        const auto packed = std::make_shared<TensorData>(packed_gate_up(gate, up));
        const auto down = std::make_shared<TensorData>(block_matrix(down_dtype, hidden, intermediate, 13 + format * 3));
        const uint64_t native_storage = QnkLinear_vulkan::storage_bytes(*packed, alignment)
                                        + QnkLinear_vulkan::storage_bytes(*down, alignment);
        block_check(native_storage != 0, "mixed MXFP4/NVFP4 projection storage sizes");
        auto operation = QnkExpert_vulkan::create(*packed, nullptr, *down, nullptr, 0.0f,
                                                  device, ExpertActivation::Silu, runtime,
                                                  OptimizationDefaultFlags);
        block_check(operation && operation->device_bytes() == native_storage,
                    "mixed packed gate/up Expert retains native GPU bytes");

        auto backend = create_vulkan_expert_backend(native_storage + 4096, device, nullptr,
                                                    runtime, OptimizationDefaultFlags);
        std::shared_ptr<const void> pin;
        block_check(backend && backend->prepare_demand(ExpertDemandRequest{"fp4-mixed-" + std::to_string(format), packed, nullptr, down, nullptr, 0, 0.0f, ExpertActivation::Silu, nullptr}, pin),
                    "mixed packed gate/up cache admission");
        block_check(pin && backend->statistics().resident_size == native_storage,
                    "mixed packed gate/up cache accounts native bytes");

        const auto input = block_input(3, hidden);
        const auto golden = block_expert_reference(*packed, nullptr, *down, input, false, 0.0f);
        ActivationBuffer direct_output;
        block_check(operation->forward(input, direct_output), "mixed packed gate/up direct Qnk Expert execution");
        block_compare_expert(direct_output, golden, "mixed packed gate/up Vulkan reference", gate_up_dtype, down_dtype);

        ActivationBuffer cached_output;
        const std::string key = "fp4-mixed-" + std::to_string(format);
        const std::array<ExpertRequest, 1> requests = {ExpertRequest{key, &input, &cached_output}};
        auto submission = backend->submit_batch(requests);
        block_check(static_cast<bool>(submission), "mixed packed gate/up cache hit submission");
        const auto results = submission->wait();
        block_check(results.size() == 1 && results[0] == ExpertStatus::Executed && submission->commit(),
                    "mixed packed gate/up cache hit executes through QnkExpert_vulkan");
        block_compare_expert(cached_output, golden, "mixed packed gate/up cache reference", gate_up_dtype, down_dtype);
    }
}

static std::shared_ptr<DeviceTensor_vulkan> block_upload(const ActivationBuffer& input, uint32_t device, const VulkanRuntimePtr& runtime)
{
    TensorData seed;
    seed.dtype = DType::Float32;
    seed.shape = {input.columns(), input.columns()};
    seed.float32_data.assign(static_cast<size_t>(input.columns()) * input.columns(), 0.f);
    auto operation = Linear::create(seed, nullptr, device, runtime, OptimizationDefaultFlags);
    block_check(static_cast<bool>(operation), "native block resident input bridge");
    auto graph = CommandGraph_vulkan::create(*operation);
    auto input_device = std::make_shared<DeviceTensor_vulkan>();
    block_check(graph && graph->upload(input, *input_device) && graph->submit() && graph->wait(),
                "native block resident input upload");
    return input_device;
}

static void block_experts(uint32_t device, DType dtype, DType down_dtype)
{
    const bool fp4 = dtype == DType::MxFp4 || dtype == DType::NvFp4;
    const uint32_t hidden = std::max<uint32_t>(64, qnk_block_size(dtype));
    const uint32_t intermediate = std::max<uint32_t>(32, qnk_block_size(down_dtype));
    const auto runtime = create_vulkan_runtime();
    const auto gate = std::make_shared<TensorData>(block_matrix(dtype, intermediate, hidden, 1));
    const auto up = std::make_shared<TensorData>(block_matrix(dtype, intermediate, hidden, 2));
    const auto down = std::make_shared<TensorData>(block_matrix(down_dtype, hidden, intermediate, 3));
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    block_check(static_cast<bool>(context), "native block Expert context");
    const uint64_t alignment = context->device()->info.buffer_offset_alignment();
    const uint64_t storage = QnkLinear_vulkan::storage_bytes(*gate, alignment)
                             + QnkLinear_vulkan::storage_bytes(*up, alignment)
                             + QnkLinear_vulkan::storage_bytes(*down, alignment);
    for (float limit : {0.0f, 0.25f})
    {
        const ExpertActivation activation = limit > 0 ? ExpertActivation::DeepSeekSwiGlu : ExpertActivation::Silu;
        auto separate = QnkExpert_vulkan::create_separate(*gate, nullptr, *up, nullptr, *down, nullptr, limit, device,
                                                          activation, runtime, OptimizationDefaultFlags);
        block_check(separate && separate->device_bytes() == storage, "separate Expert retains native bytes");
        for (size_t tokens : {size_t(1), size_t(4), size_t(5)})
        {
            const auto input = block_input(tokens, hidden);
            const auto expected = block_expert_reference(*gate, up.get(), *down, input, false, limit);
            ActivationBuffer actual;
            block_check(separate->forward(input, actual), "native host-input Expert");
            block_compare_expert(actual, expected, "Shared host reference", dtype, down_dtype);
            auto resident = block_upload(input, device, runtime);
            auto result = std::make_shared<DeviceTensor_vulkan>();
            const auto before = get_vulkan_statistics(runtime);
            block_check(separate->forward_device(*resident, *result), "resident Shared Expert");
            const auto after = get_vulkan_statistics(runtime);
            block_check(after.compute_submissions == before.compute_submissions + 1
                            && after.batch_uploads == before.batch_uploads && after.batch_downloads == before.batch_downloads,
                        "Shared chain has one submit without activation transfers");
            std::shared_ptr<DeviceTensor_vulkan> combined;
            block_check(MoeCombine_vulkan::forward({}, tokens, hidden, nullptr, result, combined),
                        "resident Shared enters device Combine");
            block_check(separate->materialize(*combined, actual), "materialize combined output");
            block_compare_expert(actual, expected, "resident Shared/Combine reference", dtype, down_dtype);
            block_check(separate->materialize(*resident, actual), "resident input remains owned");
            block_compare(actual, input, "resident input immutable");
            block_check(!separate->forward_device(*result, *result), "reject aliased device output");
            DeviceTensor_vulkan empty;
            block_check(!separate->forward_device(empty, *result) && !result->empty(),
                        "failed device call preserves prior output");
        }
    }
    if (fp4)
    {
        auto packed = std::make_shared<TensorData>(packed_gate_up(*gate, *up));
        auto operation = QnkExpert_vulkan::create(*packed, nullptr, *down, nullptr, 0.0f, device,
                                                  ExpertActivation::Silu, runtime, OptimizationDefaultFlags);
        const auto input = block_input(5, hidden);
        ActivationBuffer actual;
        block_check(operation && operation->forward(input, actual), "packed FP4 gate/up Expert");
        block_compare_expert(actual, block_expert_reference(*packed, nullptr, *down, input, false, 0.0f),
                             "packed FP4 Expert reference", dtype, down_dtype);
    }
    else
    {
        auto packed = std::make_shared<TensorData>(block_matrix(dtype, intermediate * 2, hidden, 4));
        for (bool interleaved : {false, true})
        {
            packed->qnk_interleave_rows = interleaved;
            auto operation = QnkExpert_vulkan::create(*packed, nullptr, *down, nullptr, 0.0f, device,
                                                      ExpertActivation::Silu, runtime, OptimizationDefaultFlags);
            const auto input = block_input(5, hidden);
            ActivationBuffer actual;
            block_check(operation && operation->forward(input, actual), "IQ4_NL/Q5_1 packed gate/up Expert");
            block_compare_expert(actual, block_expert_reference(*packed, nullptr, *down, input, interleaved, 0.0f),
                                 "IQ4_NL/Q5_1 packed Expert reference", dtype, down_dtype);
        }
    }
    if (fp4)
    {
        const TensorData gate_bias = fp4_bias(intermediate, 1);
        const TensorData up_bias = fp4_bias(intermediate, 2);
        const TensorData down_bias = fp4_bias(hidden, 3);
        auto biased = QnkExpert_vulkan::create_separate(*gate, &gate_bias, *up, &up_bias, *down, &down_bias,
                                                        0.0f, device, ExpertActivation::Silu,
                                                        runtime, OptimizationDefaultFlags);
        block_check(static_cast<bool>(biased), "biased FP4 Expert creation");
        const auto input = block_input(3, hidden);
        auto expected_gate = block_reference(*gate, input);
        auto expected_up = block_reference(*up, input);
        for (size_t token = 0; token < input.rows(); ++token)
            for (uint32_t row = 0; row < intermediate; ++row)
            {
                expected_gate.row(token)[row] += gate_bias.float32_data[row];
                expected_up.row(token)[row] += up_bias.float32_data[row];
            }
        ActivationBuffer activated(input.rows(), intermediate);
        for (size_t token = 0; token < input.rows(); ++token)
            for (uint32_t column = 0; column < intermediate; ++column)
            {
                const float gate_value = expected_gate.row(token)[column];
                activated.row(token)[column] = gate_value / (1.f + std::exp(-gate_value))
                                               * expected_up.row(token)[column];
            }
        auto expected = block_reference(*down, activated);
        for (size_t token = 0; token < input.rows(); ++token)
            for (uint32_t row = 0; row < hidden; ++row)
                expected.row(token)[row] += down_bias.float32_data[row];
        ActivationBuffer actual;
        block_check(biased->forward(input, actual), "biased FP4 Expert execution");
        block_compare_expert(actual, expected, "biased FP4 Expert reference", dtype, down_dtype);
    }
    auto backend = create_vulkan_expert_backend(storage + 4096, device, nullptr, runtime, OptimizationDefaultFlags);
    std::shared_ptr<const void> pin;
    block_check(backend && backend->prepare_demand(ExpertDemandRequest{"gguf-blocks", gate, nullptr, down, nullptr, 0, 0.0f, ExpertActivation::Silu, up}, pin),
                "native Expert cache admission");
    block_check(pin && backend->statistics().resident_size == storage, "cache accounts compact native bytes");
    const auto input = block_input(4, hidden);
    ActivationBuffer actual;
    const std::array<ExpertRequest, 1> requests = {ExpertRequest{"gguf-blocks", &input, &actual}};
    auto submission = backend->submit_batch(requests);
    block_check(static_cast<bool>(submission), "native Expert cache submission");
    const auto results = submission->wait();
    block_check(results.size() == 1 && results[0] == ExpertStatus::Executed && submission->commit(),
                "cached Expert executes on Vulkan");
    block_compare_expert(actual, block_expert_reference(*gate, up.get(), *down, input, false, 0.0f),
                         "cached Expert reference", dtype, down_dtype);

    const auto source = block_input(5, hidden);
    auto resident = block_upload(source, device, runtime);
    auto wave_backend = create_vulkan_expert_backend(storage * 2, device, nullptr, runtime, OptimizationDefaultFlags);
    block_check(static_cast<bool>(wave_backend), "native device Expert cache backend");
    const std::array<ExpertRoute, 3> first_routes = {{{4, 0, 0.7f}, {1, 1, 0.25f}, {4, 2, -0.1f}}};
    const std::array<ExpertRoute, 2> second_routes = {{{0, 0, 0.5f}, {1, 1, 0.4f}}};
    ActivationBuffer first_host;
    ActivationBuffer second_host;
    std::shared_ptr<DeviceTensor_vulkan> first_device;
    std::shared_ptr<DeviceTensor_vulkan> second_device;
    std::array<ExpertRequest, 2> device_requests;
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
    block_check(static_cast<bool>(cold), "device-only cold cache lookup");
    const auto cold_results = cold->wait();
    block_check(cold_results.size() == 2 && cold_results[0] == ExpertStatus::NotResident
                    && cold_results[1] == ExpertStatus::NotResident && !first_device && !second_device,
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
    block_check(upload && upload->wait(pins) == 2 && pins[0] && pins[1], "cold native Expert demand upload retains both entries");
    upload.reset();
    const auto before_device = get_vulkan_statistics(runtime);
    auto device_submission = wave_backend->submit_batch(device_requests);
    block_check(static_cast<bool>(device_submission), "device-only routed native Expert submission");
    pins = {};
    const auto device_results = device_submission->wait();
    block_check(device_results.size() == 2 && device_results[0] == ExpertStatus::Executed
                    && device_results[1] == ExpertStatus::Executed && device_submission->commit(),
                "native Expert device batch commits after upload pins release");
    device_submission.reset();
    const auto after_device = get_vulkan_statistics(runtime);
    block_check(first_device && second_device && first_host.rows() == 0 && second_host.rows() == 0
                    && after_device.compute_submissions == before_device.compute_submissions + 1
                    && after_device.batch_uploads == before_device.batch_uploads
                    && after_device.batch_downloads == before_device.batch_downloads,
                "routed native GPU gather and Expert batch perform no activation transfer or CPU output");
    const auto reference_for = [&](std::span<const ExpertRoute> routes) {
        ActivationBuffer gathered(routes.size(), hidden);
        for (size_t row = 0; row < routes.size(); ++row)
            std::copy_n(source.row(routes[row].token_index), hidden, gathered.row(row));
        return block_expert_reference(*gate, up.get(), *down, gathered, false, 0.0f);
    };
    const auto first_expected = reference_for(first_routes);
    const auto second_expected = reference_for(second_routes);
    block_check(MoeCombine_vulkan::materialize(*first_device, actual), "native routed first output materialization");
    block_compare_expert(actual, first_expected, "native GPU gather handles permuted and repeated source rows", dtype, down_dtype);
    const std::array<MoeCombineInput_vulkan, 2> combine_inputs = {{{nullptr, first_device, first_routes}, {nullptr, second_device, second_routes}}};
    std::shared_ptr<DeviceTensor_vulkan> combined;
    block_check(MoeCombine_vulkan::forward(combine_inputs, source.rows(), hidden, nullptr, {}, combined)
                    && MoeCombine_vulkan::materialize(*combined, actual),
                "native routed results enter weighted GPU Combine");
    ActivationBuffer combined_expected(source.rows(), hidden);
    for (size_t row = 0; row < first_routes.size(); ++row)
        for (uint32_t column = 0; column < hidden; ++column)
            combined_expected.row(first_routes[row].token_index)[column] += first_routes[row].weight * first_expected.row(row)[column];
    for (size_t row = 0; row < second_routes.size(); ++row)
        for (uint32_t column = 0; column < hidden; ++column)
            combined_expected.row(second_routes[row].token_index)[column] += second_routes[row].weight * second_expected.row(row)[column];
    block_compare_expert(actual, combined_expected, "native GPU weighted route reduction reference", dtype, down_dtype);
    std::shared_ptr<const void> replacement_pin;
    block_check(wave_backend->prepare_demand(ExpertDemandRequest{"native-device-replacement", gate, nullptr, down, nullptr,
                                                                 0, 0.0f, ExpertActivation::Silu, up},
                                             replacement_pin)
                    && wave_backend->statistics().evictions != 0,
                "native cold demand evicts a completed entry");
    block_check(MoeCombine_vulkan::materialize(*first_device, actual), "completed native activation survives weight cache eviction");
    block_compare_expert(actual, first_expected, "native output survives weight cache eviction", dtype, down_dtype);
    const std::array<ExpertRoute, 1> invalid_routes = {{{5, 0, 1.f}}};
    ExpertRequest invalid_request = device_requests[0];
    invalid_request.key = "native-device-replacement";
    invalid_request.device_routes = invalid_routes;
    invalid_request.input_rows = invalid_routes.size();
    const auto preserved_output = first_device;
    auto invalid_submission = wave_backend->submit_batch({&invalid_request, 1});
    block_check(static_cast<bool>(invalid_submission), "malformed device route submission is contained");
    const auto invalid_results = invalid_submission->wait();
    invalid_submission->abort();
    block_check(invalid_results.size() == 1 && invalid_results[0] == ExpertStatus::Failed
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
            for (ncnn::moe::DType dtype : {ncnn::moe::DType::IQ4NL, ncnn::moe::DType::Q5_1,
                                           ncnn::moe::DType::Q4_0, ncnn::moe::DType::Q4_1,
                                           ncnn::moe::DType::Q5_0, ncnn::moe::DType::IQ4XS})
            {
                ncnn::moe::block_projections(device, dtype);
                ncnn::moe::block_experts(device, dtype, dtype);
            }
            for (ncnn::moe::DType dtype : {ncnn::moe::DType::MxFp4, ncnn::moe::DType::NvFp4})
            {
                ncnn::moe::block_projections(device, dtype);
                ncnn::moe::block_experts(device, dtype, dtype);
            }
            ncnn::moe::mixed_fp4_experts(device);
            const std::array<std::array<ncnn::moe::DType, 2>, 4> mixed_formats = {{
                {ncnn::moe::DType::Q4_0, ncnn::moe::DType::Q4_1},
                {ncnn::moe::DType::Q4_1, ncnn::moe::DType::Q5_0},
                {ncnn::moe::DType::Q5_0, ncnn::moe::DType::IQ4XS},
                {ncnn::moe::DType::IQ4XS, ncnn::moe::DType::Q4_0},
            }};
            for (const auto& formats : mixed_formats)
                ncnn::moe::block_experts(device, formats[0], formats[1]);
            // Retain older K-quant and 32-value mixed-block references.
            ncnn::moe::block_experts(device, ncnn::moe::DType::Q4K, ncnn::moe::DType::Q5_1);
            ncnn::moe::block_experts(device, ncnn::moe::DType::Q8_0, ncnn::moe::DType::Q5_1);
        }
        std::cout << "Native GGUF block Vulkan projection, resident Shared chain and Expert cache tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Native GGUF block Vulkan test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
