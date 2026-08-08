#include "cpu_qnk.h"

#include "cpu_mxfp4.h"
#include "cpu_qnk_msvc.h"
#include "cpu_batch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>

namespace ncnn {
namespace moe {

static float half_to_float(uint16_t value) noexcept
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = sign;
    if (exponent == 0)
    {
        if (mantissa != 0)
        {
            uint32_t normalized = mantissa;
            int32_t exponent_shift = -1;
            while ((normalized & 0x0400u) == 0)
            {
                normalized <<= 1;
                --exponent_shift;
            }
            normalized &= 0x03ffu;
            bits |= static_cast<uint32_t>(127 - 15 + 1 + exponent_shift) << 23;
            bits |= normalized << 13;
        }
    }
    else if (exponent == 0x1fu)
    {
        bits |= 0x7f800000u | (mantissa << 13);
    }
    else
    {
        bits |= (exponent + (127 - 15)) << 23;
        bits |= mantissa << 13;
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t load_u16(const uint8_t* source) noexcept
{
    uint16_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

static float load_f16(const uint8_t* source) noexcept
{
    return half_to_float(load_u16(source));
}

static void get_scale_min_k4(
    int index,
    const uint8_t* scales,
    uint8_t& scale,
    uint8_t& minimum) noexcept
{
    if (index < 4)
    {
        scale = scales[index] & 63u;
        minimum = scales[index + 4] & 63u;
    }
    else
    {
        scale = static_cast<uint8_t>((scales[index + 4] & 0x0fu) | ((scales[index - 4] >> 6) << 4));
        minimum = static_cast<uint8_t>((scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
    }
}

static void dequantize_q2(const uint8_t* block, float* output) noexcept
{
    const float d = load_f16(block + 80);
    const float minimum = load_f16(block + 82);
    const uint8_t* scales = block;
    const uint8_t* q = block + 16;
    int scale_index = 0;
    float* destination = output;
    for (int n = 0; n < 256; n += 128)
    {
        int shift = 0;
        for (int group = 0; group < 4; ++group)
        {
            const uint8_t scale_min = scales[scale_index++];
            const float dl = d * (scale_min & 0x0fu);
            const float ml = minimum * (scale_min >> 4);
            for (int lane = 0; lane < 16; ++lane)
                destination[lane] = dl * static_cast<float>((q[lane] >> shift) & 3u) - ml;
            destination += 16;

            const uint8_t second_scale_min = scales[scale_index++];
            const float second_dl = d * (second_scale_min & 0x0fu);
            const float second_ml = minimum * (second_scale_min >> 4);
            for (int lane = 0; lane < 16; ++lane)
                destination[lane] = second_dl * static_cast<float>((q[lane + 16] >> shift) & 3u) - second_ml;
            destination += 16;
            shift += 2;
        }
        q += 32;
    }
}

static void dequantize_q3(const uint8_t* block, float* output) noexcept
{
    constexpr uint32_t kmask1 = 0x03030303u;
    constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t auxiliary[4] = {};
    std::memcpy(auxiliary, block + 96, 12);
    const uint32_t temporary = auxiliary[2];
    auxiliary[2] = ((auxiliary[0] >> 4) & kmask2) | (((temporary >> 4) & kmask1) << 4);
    auxiliary[3] = ((auxiliary[1] >> 4) & kmask2) | (((temporary >> 6) & kmask1) << 4);
    auxiliary[0] = (auxiliary[0] & kmask2) | (((temporary >> 0) & kmask1) << 4);
    auxiliary[1] = (auxiliary[1] & kmask2) | (((temporary >> 2) & kmask1) << 4);
    int8_t scales[16] = {};
    std::memcpy(scales, auxiliary, sizeof(scales));

    const float d = load_f16(block + 108);
    const uint8_t* q = block + 32;
    const uint8_t* high_bits = block;
    uint8_t mask = 1;
    int scale_index = 0;
    float* destination = output;
    for (int n = 0; n < 256; n += 128)
    {
        int shift = 0;
        for (int group = 0; group < 4; ++group)
        {
            const float scale = d * static_cast<float>(scales[scale_index++] - 32);
            for (int lane = 0; lane < 16; ++lane)
            {
                const int high = (high_bits[lane] & mask) != 0 ? 0 : 4;
                destination[lane] = scale * (static_cast<int>((q[lane] >> shift) & 3u) - high);
            }
            destination += 16;

            const float second_scale = d * static_cast<float>(scales[scale_index++] - 32);
            for (int lane = 0; lane < 16; ++lane)
            {
                const int high = (high_bits[lane + 16] & mask) != 0 ? 0 : 4;
                destination[lane] = second_scale * (static_cast<int>((q[lane + 16] >> shift) & 3u) - high);
            }
            destination += 16;
            shift += 2;
            mask = static_cast<uint8_t>(mask << 1);
        }
        q += 32;
    }
}

static void dequantize_q4(const uint8_t* block, float* output) noexcept
{
    const float d = load_f16(block);
    const float minimum = load_f16(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* q = block + 16;
    int scale_index = 0;
    float* destination = output;
    for (int n = 0; n < 256; n += 64)
    {
        uint8_t scale_value = 0;
        uint8_t min_value = 0;
        get_scale_min_k4(scale_index++, scales, scale_value, min_value);
        const float first_scale = d * scale_value;
        const float first_minimum = minimum * min_value;
        get_scale_min_k4(scale_index++, scales, scale_value, min_value);
        const float second_scale = d * scale_value;
        const float second_minimum = minimum * min_value;
        for (int lane = 0; lane < 32; ++lane)
            destination[lane] = first_scale * static_cast<float>(q[lane] & 0x0fu) - first_minimum;
        for (int lane = 0; lane < 32; ++lane)
            destination[32 + lane] = second_scale * static_cast<float>(q[lane] >> 4) - second_minimum;
        destination += 64;
        q += 32;
    }
}

static void dequantize_q5(const uint8_t* block, float* output) noexcept
{
    const float d = load_f16(block);
    const float minimum = load_f16(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* high_bits = block + 16;
    const uint8_t* low_bits = block + 48;
    int scale_index = 0;
    uint8_t first_mask = 1;
    uint8_t second_mask = 2;
    float* destination = output;
    for (int n = 0; n < 256; n += 64)
    {
        uint8_t scale_value = 0;
        uint8_t min_value = 0;
        get_scale_min_k4(scale_index++, scales, scale_value, min_value);
        const float first_scale = d * scale_value;
        const float first_minimum = minimum * min_value;
        get_scale_min_k4(scale_index++, scales, scale_value, min_value);
        const float second_scale = d * scale_value;
        const float second_minimum = minimum * min_value;
        for (int lane = 0; lane < 32; ++lane)
        {
            const int high = (high_bits[lane] & first_mask) != 0 ? 16 : 0;
            destination[lane] = first_scale * static_cast<float>((low_bits[lane] & 0x0fu) + high) - first_minimum;
        }
        for (int lane = 0; lane < 32; ++lane)
        {
            const int high = (high_bits[lane] & second_mask) != 0 ? 16 : 0;
            destination[32 + lane] = second_scale * static_cast<float>((low_bits[lane] >> 4) + high) - second_minimum;
        }
        destination += 64;
        low_bits += 32;
        scale_index += 0;
        first_mask = static_cast<uint8_t>(first_mask << 2);
        second_mask = static_cast<uint8_t>(second_mask << 2);
    }
}

static void dequantize_q6(const uint8_t* block, float* output) noexcept
{
    const float d = load_f16(block + 208);
    const uint8_t* low_bits = block;
    const uint8_t* high_bits = block + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
    float* destination = output;
    for (int n = 0; n < 256; n += 128)
    {
        for (int lane = 0; lane < 32; ++lane)
        {
            const int q1 = static_cast<int>((low_bits[lane] & 0x0fu) | (((high_bits[lane] >> 0) & 3u) << 4)) - 32;
            const int q2 = static_cast<int>((low_bits[lane + 32] & 0x0fu) | (((high_bits[lane] >> 2) & 3u) << 4)) - 32;
            const int q3 = static_cast<int>((low_bits[lane] >> 4) | (((high_bits[lane] >> 4) & 3u) << 4)) - 32;
            const int q4 = static_cast<int>((low_bits[lane + 32] >> 4) | (((high_bits[lane] >> 6) & 3u) << 4)) - 32;
            destination[lane] = d * scales[0] * q1;
            destination[lane + 32] = d * scales[2] * q2;
            destination[lane + 64] = d * scales[4] * q3;
            destination[lane + 96] = d * scales[6] * q4;
        }
        destination += 128;
        low_bits += 64;
        high_bits += 32;
        scales += 8;
    }
}

static void dequantize_q8(const uint8_t* block, float* output) noexcept
{
    float scale = 0.0f;
    std::memcpy(&scale, block, sizeof(scale));
    const int8_t* values = reinterpret_cast<const int8_t*>(block + 4);
    for (uint32_t index = 0; index < qnk_block_elements; ++index)
        output[index] = scale * static_cast<float>(values[index]);
}

static void qnk_gemm_scalar(
    const QnKPack& weights,
    const float* input,
    size_t input_stride,
    size_t token_count,
    float* output,
    size_t output_stride) noexcept
{
    for (size_t row = 0; row < weights.rows; ++row)
    {
        for (size_t token = 0; token < token_count; ++token)
        {
            const float* token_input = input + token * input_stride;
            float sum = 0.0f;
            for (uint32_t block = 0; block < weights.block_count; ++block)
                sum += qnk_dot_block(weights.dtype, qnk_packed_block(weights, row, block), token_input + static_cast<size_t>(block) * qnk_block_elements);
            output[token * output_stride + row] = sum;
        }
    }
}

static void scalar_qnk_q8k_quantize(
    const float* source,
    uint8_t* output,
    uint32_t columns) noexcept
{
    const uint32_t block_count = columns / qnk_block_elements;
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const float* values = source + static_cast<size_t>(block_index) * qnk_block_elements;
        uint8_t* encoded = output + static_cast<size_t>(block_index) * qnk_block_bytes(DType::Q8K);
        float maximum = 0.0f;
        float signed_maximum = 0.0f;
        for (uint32_t index = 0; index < qnk_block_elements; ++index)
        {
            const float absolute = std::fabs(values[index]);
            if (absolute > maximum)
            {
                maximum = absolute;
                signed_maximum = values[index];
            }
        }
        if (maximum == 0.0f)
        {
            std::memset(encoded, 0, qnk_block_bytes(DType::Q8K));
            continue;
        }
        const float inverse_scale = -127.0f / signed_maximum;
        const float scale = 1.0f / inverse_scale;
        std::memcpy(encoded, &scale, sizeof(scale));
        int8_t* quantized = reinterpret_cast<int8_t*>(encoded + 4);
        for (uint32_t index = 0; index < qnk_block_elements; ++index)
        {
            const long rounded = std::lrintf(inverse_scale * values[index]);
            quantized[index] = static_cast<int8_t>(std::clamp<long>(rounded, -127, 127));
        }
        for (uint32_t sum_index = 0; sum_index < qnk_block_elements / 16; ++sum_index)
        {
            int16_t sum = 0;
            for (uint32_t lane = 0; lane < 16; ++lane)
                sum = static_cast<int16_t>(sum + quantized[sum_index * 16 + lane]);
            std::memcpy(encoded + 260 + sum_index * sizeof(int16_t), &sum, sizeof(sum));
        }
    }
}

static std::shared_ptr<const QnKPack> get_qnk_packed_weights(
    const TensorData& matrix,
    size_t rows,
    uint32_t columns) noexcept
{
    static std::mutex build_locks[64];
    const std::span<const uint8_t> raw = matrix.qnk_values();
    const uintptr_t storage_key = reinterpret_cast<uintptr_t>(raw.data());
    std::lock_guard<std::mutex> lock(build_locks[(storage_key >> 6) & 63u]);
    std::shared_ptr<const QnKPack> cached = matrix.qnk_packed;
    if (cached && cached->valid() && cached->dtype == matrix.dtype && cached->rows == rows && cached->columns == columns)
        return cached;

    auto packed = std::make_shared<QnKPack>();
    if (!qnk_pack_weights(raw.data(), raw.size(), matrix.dtype, rows, columns, *packed))
        return {};
    std::shared_ptr<const QnKPack> desired = packed;
    matrix.qnk_packed = desired;
    return desired;
}

size_t qnk_block_bytes(DType dtype) noexcept
{
    switch (dtype)
    {
    case DType::Q2K: return 84;
    case DType::Q3K: return 110;
    case DType::Q4K: return 144;
    case DType::Q5K: return 176;
    case DType::Q6K: return 210;
    case DType::Q8K: return 292;
    default: return 0;
    }
}

uint64_t qnk_storage_bytes(DType dtype, size_t rows, uint32_t columns) noexcept
{
    if (!qnk_shape_supported(dtype, rows, columns))
        return 0;
    const uint64_t row_blocks = static_cast<uint64_t>(columns / qnk_block_elements);
    const uint64_t bytes_per_block = qnk_block_bytes(dtype);
    if (row_blocks != 0 && bytes_per_block > std::numeric_limits<uint64_t>::max() / row_blocks)
        return 0;
    const uint64_t row_bytes = row_blocks * bytes_per_block;
    if (rows != 0 && row_bytes > std::numeric_limits<uint64_t>::max() / rows)
        return 0;
    return row_bytes * rows;
}

bool qnk_shape_supported(DType dtype, size_t rows, uint32_t columns) noexcept
{
    return is_qnk_dtype(dtype) && rows != 0 && columns != 0 && columns % qnk_block_elements == 0
           && qnk_block_bytes(dtype) != 0;
}

void qnk_dequantize_block(DType dtype, const uint8_t* block, float* output) noexcept
{
    if (!block || !output)
        return;
    switch (dtype)
    {
    case DType::Q2K: dequantize_q2(block, output); break;
    case DType::Q3K: dequantize_q3(block, output); break;
    case DType::Q4K: dequantize_q4(block, output); break;
    case DType::Q5K: dequantize_q5(block, output); break;
    case DType::Q6K: dequantize_q6(block, output); break;
    case DType::Q8K: dequantize_q8(block, output); break;
    default: std::fill_n(output, qnk_block_elements, 0.0f); break;
    }
}

float qnk_dot_block(DType dtype, const uint8_t* block, const float* input) noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if ((dtype == DType::Q2K || dtype == DType::Q3K || dtype == DType::Q4K
         || dtype == DType::Q5K || dtype == DType::Q6K || dtype == DType::Q8K)
        && (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2
            || mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512))
    {
        return msvc_avx2_qnk_dot_block(dtype, block, input);
    }
#endif
    alignas(64) float decoded[qnk_block_elements];
    qnk_dequantize_block(dtype, block, decoded);
    float sum = 0.0f;
    for (uint32_t index = 0; index < qnk_block_elements; ++index)
        sum += decoded[index] * input[index];
    return sum;
}

void qnk_q8k_quantize(const float* source, uint8_t* output, uint32_t columns) noexcept
{
    if (!source || !output || columns == 0 || columns % qnk_block_elements != 0)
        return;
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512)
    {
        msvc_avx512_qnk_q8k_quantize(source, output, columns);
        return;
    }
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2)
    {
        msvc_avx2_qnk_q8k_quantize(source, output, columns);
        return;
    }
#endif
    scalar_qnk_q8k_quantize(source, output, columns);
}

void qnk_q8k_quantize_batch(
    const float* source,
    size_t input_stride,
    size_t rows,
    uint32_t columns,
    std::vector<uint8_t>& output) noexcept
{
    if (!source || rows == 0 || !qnk_shape_supported(DType::Q8K, rows, columns))
    {
        output.clear();
        return;
    }
    const size_t row_bytes = static_cast<size_t>(qnk_storage_bytes(DType::Q8K, 1, columns));
    output.resize(rows * row_bytes);
    for (size_t row = 0; row < rows; ++row)
        qnk_q8k_quantize(source + row * input_stride, output.data() + row * row_bytes, columns);
}

bool qnk_pack_weights(
    const uint8_t* raw,
    size_t raw_bytes,
    DType dtype,
    size_t rows,
    uint32_t columns,
    QnKPack& output) noexcept
{
    const uint64_t expected = qnk_storage_bytes(dtype, rows, columns);
    if (!raw || expected == 0 || expected != raw_bytes || rows > std::numeric_limits<size_t>::max() - 7)
        return false;
    const uint32_t block_count = columns / qnk_block_elements;
    const size_t tile_count = (rows + 7) / 8;
    const size_t block_bytes = qnk_block_bytes(dtype);
    if (tile_count > std::numeric_limits<size_t>::max() / block_count
        || tile_count * block_count > std::numeric_limits<size_t>::max() / (8 * block_bytes))
    {
        return false;
    }
    QnKPack packed;
    packed.dtype = dtype;
    packed.rows = rows;
    packed.columns = columns;
    packed.block_count = block_count;
    packed.tile_rows = 8;
    packed.storage.resize(tile_count * static_cast<size_t>(block_count) * 8 * block_bytes, 0);
    for (size_t tile = 0; tile < tile_count; ++tile)
    {
        for (uint32_t block = 0; block < block_count; ++block)
        {
            for (uint32_t lane = 0; lane < 8; ++lane)
            {
                const size_t row = tile * 8 + lane;
                uint8_t* destination = packed.storage.data()
                                       + (((tile * block_count + block) * 8 + lane) * block_bytes);
                if (row < rows)
                {
                    const uint8_t* source = raw
                                             + ((row * static_cast<size_t>(block_count) + block) * block_bytes);
                    std::memcpy(destination, source, block_bytes);
                }
            }
        }
    }
    output = std::move(packed);
    return true;
}

bool qnk_linear_batch_into(const TensorData& matrix, const CpuBatch& input, CpuBatch& output) noexcept
{
    if (!is_qnk_dtype(matrix.dtype) || matrix.shape.size() != 2)
        return false;
    const size_t rows = matrix.shape[0];
    const uint32_t columns = matrix.shape[1];
    if (!qnk_shape_supported(matrix.dtype, rows, columns) || input.columns() != columns)
        return false;
    const std::span<const uint8_t> raw = matrix.qnk_values();
    if (raw.size() != qnk_storage_bytes(matrix.dtype, rows, columns))
        return false;
    const std::shared_ptr<const QnKPack> packed = get_qnk_packed_weights(matrix, rows, columns);
    if (!packed)
        return false;

    output.reset(input.rows(), static_cast<uint32_t>(rows), false);
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512)
    {
        msvc_avx512_qnk_gemm(*packed, input.row(0), columns, input.rows(), output.row(0), output.columns());
        return true;
    }
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2)
    {
        msvc_avx2_qnk_gemm(*packed, input.row(0), columns, input.rows(), output.row(0), output.columns());
        return true;
    }
#endif
    qnk_gemm_scalar(*packed, input.row(0), columns, input.rows(), output.row(0), output.columns());
    return true;
}

} // namespace moe
} // namespace ncnn
