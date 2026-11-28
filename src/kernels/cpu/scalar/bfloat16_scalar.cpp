#include "kernels/cpu/scalar/scalar_internal.h"

#include <cstring>

namespace ncnn {
namespace moe {

static void scalar_float_to_bfloat16_array(uint16_t* output,
                                           const float* input,
                                           uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, input + index, sizeof(bits));
        const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
        output[index] = static_cast<uint16_t>((bits + rounding) >> 16);
    }
}

static float scalar_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    float sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(weights[index]) << 16;
        float weight = 0.0f;
        std::memcpy(&weight, &bits, sizeof(weight));
        sum += weight * input[index];
    }
    return sum;
}

static void scalar_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(input[index]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        output[index] += scale * value;
    }
}

static float scalar_bfloat16_pair_dot(const uint16_t* left,
                                      const uint16_t* right,
                                      uint32_t count) noexcept
{
    float sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
    {
        const uint32_t left_bits = static_cast<uint32_t>(left[index]) << 16;
        const uint32_t right_bits = static_cast<uint32_t>(right[index]) << 16;
        float left_value = 0.0f;
        float right_value = 0.0f;
        std::memcpy(&left_value, &left_bits, sizeof(left_value));
        std::memcpy(&right_value, &right_bits, sizeof(right_value));
        sum += left_value * right_value;
    }
    return sum;
}

static void scalar_pv_bf16(float* output,
                           const uint16_t* values0,
                           size_t value_stride,
                           const float* weights0,
                           uint32_t count0,
                           const uint16_t* values1,
                           const float* weights1,
                           uint32_t count1,
                           uint32_t columns) noexcept
{
    for (uint32_t key = 0; key < count0; ++key)
    {
        const float weight = weights0[key];
        const uint16_t* row = values0 + static_cast<size_t>(key) * value_stride;
        for (uint32_t column = 0; column < columns; ++column)
        {
            const uint32_t bits = static_cast<uint32_t>(row[column]) << 16;
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            output[column] += weight * value;
        }
    }
    for (uint32_t key = 0; key < count1; ++key)
    {
        const float weight = weights1[key];
        const uint16_t* row = values1 + static_cast<size_t>(key) * value_stride;
        for (uint32_t column = 0; column < columns; ++column)
        {
            const uint32_t bits = static_cast<uint32_t>(row[column]) << 16;
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            output[column] += weight * value;
        }
    }
}

void register_scalar_bfloat16_ops(CpuKernelOps& ops, CpuBackendCaps& caps) noexcept
{
    ops.bfloat16_dot = scalar_bfloat16_dot;
    ops.float_to_bfloat16_array = scalar_float_to_bfloat16_array;
    ops.bfloat16_pair_dot = scalar_bfloat16_pair_dot;
    ops.bfloat16_scaled_add = scalar_bfloat16_scaled_add;
    ops.pv_bf16 = scalar_pv_bf16;
    caps.bfloat16_dot_name = "scalar-fp32";
    caps.bf16_fast_dp = false;
}

} // namespace moe
} // namespace ncnn
