#include "kernels/cpu/scalar/scalar_internal.h"

#include "kernels/float8.h"

#include <algorithm>
#include <cmath>

namespace ncnn {
namespace moe {

static float scalar_float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept
{
    float result = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        float partial = 0.0f;
        for (uint32_t index = block_begin; index < block_end; ++index)
            partial += float8_e4m3_to_float(weights[index]) * input[index];
        result += partial * scales[block];
    }
    return result;
}

static void scalar_float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count,
                                               uint32_t block_size, uint32_t row_count, float* output) noexcept
{
    for (uint32_t row = 0; row < row_count; ++row)
    {
        output[row] = scalar_float8_e4m3_block_dot(weights + static_cast<size_t>(row) * weight_row_stride, scales, input, count, block_size);
    }
}

static void scalar_quantize_float8_e4m3(const float* source,
                                        float* values,
                                        uint32_t count,
                                        uint32_t block_size,
                                        bool power_of_two_scale) noexcept
{
    for (uint32_t block_begin = 0; block_begin < count; block_begin += block_size)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        float maximum = 1e-4f;
        for (uint32_t index = block_begin; index < block_end; ++index)
            maximum = std::max(maximum, std::fabs(source[index]));
        float scale = maximum / 448.0f;
        if (power_of_two_scale)
            scale = std::exp2(std::ceil(std::log2(scale)));
        for (uint32_t index = block_begin; index < block_end; ++index)
        {
            const float normalized = std::clamp(source[index] / scale, -448.0f, 448.0f);
            values[index] = float8_e4m3_to_float(float_to_float8_e4m3(normalized)) * scale;
        }
    }
}

void register_scalar_float8_ops(CpuKernelOps& ops, CpuBackendCaps& caps) noexcept
{
    ops.float8_e4m3_block_dot = scalar_float8_e4m3_block_dot;
    ops.float8_e4m3_block_dot_rows4 = scalar_float8_e4m3_block_dot_rows4;
    ops.quantize_float8_e4m3 = scalar_quantize_float8_e4m3;
    caps.float8_name = "scalar";
    caps.float8_native = false;
}

} // namespace moe
} // namespace ncnn
