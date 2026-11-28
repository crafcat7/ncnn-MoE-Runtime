#include "kernels/cpu/cpu_ops.h"

#include "kernels/cpu/gemm_tile.h"
#include "kernels/cpu/scalar/scalar_internal.h"
#include "kernels/fastmath.h"
#include "kernels/vector.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace ncnn {
namespace moe {

// Portable half conversion for scalar tails and non-ARM builds.
float f16_to_f32(uint16_t value) noexcept
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1Fu;
    uint32_t mantissa = value & 0x3FFu;
    uint32_t bits = 0;
    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
        }
        else
        {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0)
            {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    }
    else if (exponent == 0x1Fu)
    {
        bits = sign | 0x7F800000u | (mantissa << 13);
    }
    else
    {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(bits);
}

uint16_t float_to_f16(float value) noexcept
{
    // Saturate so KV outliers cannot produce inf/NaN.
    if (value > 65504.0f)
        value = 65504.0f;
    else if (value < -65504.0f)
        value = -65504.0f;
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t raw_exponent = (bits >> 23) & 0xFFu;
    uint32_t mantissa = bits & 0x7FFFFFu;
    int32_t exponent = static_cast<int32_t>(raw_exponent) - 127 + 15;
    if (raw_exponent == 0xFFu)
        return static_cast<uint16_t>(sign | (mantissa != 0 ? 0x7E00u : 0x7C00u));
    if (exponent >= 0x1F)
        return static_cast<uint16_t>(sign | 0x7C00u);
    if (exponent <= 0)
    {
        if (exponent < -10)
            return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const int32_t shift = 14 - exponent;
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1u);
        const uint32_t halfway = UINT32_C(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0))
            ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t rounded = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u) != 0))
    {
        ++rounded;
        if (rounded == 0x400u)
        {
            rounded = 0;
            ++exponent;
            if (exponent >= 0x1F)
                return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | rounded);
}

static float scalar_float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    float result = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        result += left[index] * right[index];
    return result;
}

static size_t scalar_float_argmax(const float* values, size_t count) noexcept
{
    if (count == 0)
        return 0;
    size_t selected = count;
    float selected_value = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < count; ++index)
    {
        const float value = values[index];
        if (value > selected_value && std::isfinite(value))
        {
            selected = index;
            selected_value = value;
        }
    }
    return selected;
}

static void scalar_float_gemm_tile_4x4(const float* weights,
                                       size_t weight_stride,
                                       const float* input,
                                       size_t input_stride,
                                       uint32_t input_columns,
                                       uint32_t output_count,
                                       uint32_t token_count,
                                       float* output,
                                       size_t output_stride) noexcept
{
    if (token_count == 0 || output_count == 0)
        return;
    assert(token_count <= 4 && output_count <= 4);
    if (token_count > 4 || output_count > 4)
        std::abort();
    float accumulators[4][4] = {};
    for (uint32_t column = 0; column < input_columns; ++column)
    {
        for (uint32_t token = 0; token < token_count; ++token)
        {
            const float value = input[static_cast<size_t>(token) * input_stride + column];
            for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            {
                accumulators[token][output_index] += value * weights[static_cast<size_t>(output_index) * weight_stride + column];
            }
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            output[static_cast<size_t>(token) * output_stride + output_index] = accumulators[token][output_index];
}

static void scalar_float_gemm_tile_4x8(const float* weights,
                                       size_t weight_stride,
                                       const float* input,
                                       size_t input_stride,
                                       uint32_t input_columns,
                                       uint32_t output_count,
                                       uint32_t token_count,
                                       float* output,
                                       size_t output_stride) noexcept
{
    if (token_count == 0 || output_count == 0)
        return;
    assert(token_count <= 4 && output_count <= 8);
    if (token_count > 4 || output_count > 8)
        std::abort();
    float accumulators[4][8] = {};
    for (uint32_t column = 0; column < input_columns; ++column)
    {
        float input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
            input_values[token] = input[static_cast<size_t>(token) * input_stride + column];
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const float weight = weights[static_cast<size_t>(output_index) * weight_stride + column];
            for (uint32_t token = 0; token < token_count; ++token)
                accumulators[token][output_index] += input_values[token] * weight;
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            output[static_cast<size_t>(token) * output_stride + output_index] = accumulators[token][output_index];
}

static void scalar_bfloat16_gemm_tile_4x8(const uint16_t* weights,
                                          size_t weight_stride,
                                          const float* input,
                                          size_t input_stride,
                                          uint32_t input_columns,
                                          uint32_t output_count,
                                          uint32_t token_count,
                                          float* output,
                                          size_t output_stride) noexcept
{
    if (token_count == 0 || output_count == 0)
        return;
    assert(token_count <= 4 && output_count <= 8);
    if (token_count > 4 || output_count > 8)
        std::abort();
    float accumulators[4][8] = {};
    for (uint32_t column = 0; column < input_columns; ++column)
    {
        float input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
            input_values[token] = input[static_cast<size_t>(token) * input_stride + column];
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const uint32_t bits = static_cast<uint32_t>(weights[static_cast<size_t>(output_index) * weight_stride + column])
                                  << 16;
            const float weight = std::bit_cast<float>(bits);
            for (uint32_t token = 0; token < token_count; ++token)
                accumulators[token][output_index] += input_values[token] * weight;
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            output[static_cast<size_t>(token) * output_stride + output_index] = accumulators[token][output_index];
}

static void scalar_float_gemm_4x4(const float* weights,
                                  size_t weight_stride,
                                  const float* input,
                                  size_t input_stride,
                                  uint32_t input_columns,
                                  uint32_t output_count,
                                  uint32_t token_count,
                                  float* output,
                                  size_t output_stride) noexcept
{
    run_tiled_float_gemm(4u, 4u, scalar_float_gemm_tile_4x4, weights, weight_stride, input, input_stride,
                         input_columns, output_count, token_count, output, output_stride);
}

static void scalar_float_gemm_4x8(const float* weights,
                                  size_t weight_stride,
                                  const float* input,
                                  size_t input_stride,
                                  uint32_t input_columns,
                                  uint32_t output_count,
                                  uint32_t token_count,
                                  float* output,
                                  size_t output_stride) noexcept
{
    run_tiled_float_gemm(4u, 8u, scalar_float_gemm_tile_4x8, weights, weight_stride, input, input_stride,
                         input_columns, output_count, token_count, output, output_stride);
}

static void scalar_bfloat16_gemm_4x8(const uint16_t* weights,
                                     size_t weight_stride,
                                     const float* input,
                                     size_t input_stride,
                                     uint32_t input_columns,
                                     uint32_t output_count,
                                     uint32_t token_count,
                                     float* output,
                                     size_t output_stride) noexcept
{
    run_tiled_bfloat16_gemm(4u, 8u, scalar_bfloat16_gemm_tile_4x8, weights, weight_stride, input, input_stride,
                            input_columns, output_count, token_count, output, output_stride);
}

static void scalar_f32_to_f16(uint16_t* output, const float* input, size_t count) noexcept
{
    for (size_t index = 0; index < count; ++index)
        output[index] = float_to_f16(input[index]);
}

static void scalar_bf16_to_f16(uint16_t* output, const uint16_t* input, size_t count) noexcept
{
    for (size_t index = 0; index < count; ++index)
    {
        const float value = std::bit_cast<float>(static_cast<uint32_t>(input[index]) << 16);
        output[index] = float_to_f16(value);
    }
}

static void scalar_f16_gemm_4x8(const uint16_t* weights,
                                size_t weight_stride,
                                const uint16_t* input,
                                size_t input_stride,
                                uint32_t input_columns,
                                uint32_t output_count,
                                uint32_t token_count,
                                float* output,
                                size_t output_stride) noexcept
{
    for (uint32_t row_base = 0; row_base < token_count; row_base += 4)
    {
        const uint32_t rows = std::min(4u, token_count - row_base);
        for (uint32_t column_base = 0; column_base < output_count; column_base += 4)
        {
            const uint32_t columns = std::min(4u, output_count - column_base);
            for (uint32_t row = 0; row < rows; ++row)
            {
                for (uint32_t column = 0; column < columns; ++column)
                {
                    float sum = 0.0f;
                    for (uint32_t offset = 0; offset < input_columns; ++offset)
                    {
                        sum += f16_to_f32(input[(row_base + row) * input_stride + offset])
                               * f16_to_f32(weights[(column_base + column) * weight_stride + offset]);
                    }
                    output[(row_base + row) * output_stride + column_base + column] = sum;
                }
            }
        }
    }
}

static void scalar_f16_pv(float* output,
                          const uint16_t* values,
                          size_t value_stride,
                          const uint16_t* weights,
                          uint32_t count,
                          uint32_t columns) noexcept
{
    for (uint32_t column = 0; column < columns; ++column)
    {
        float sum = 0.0f;
        for (uint32_t index = 0; index < count; ++index)
        {
            sum += f16_to_f32(values[static_cast<size_t>(index) * value_stride + column])
                   * f16_to_f32(weights[index]);
        }
        output[column] = sum;
    }
}

static void scalar_f16_pv4(float* output,
                           const uint16_t* values,
                           size_t value_stride,
                           const float* weights0,
                           const float* weights1,
                           const float* weights2,
                           const float* weights3,
                           uint32_t count,
                           uint32_t columns) noexcept
{
    for (uint32_t column = 0; column < columns; ++column)
    {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (uint32_t index = 0; index < count; ++index)
        {
            const float value = f16_to_f32(values[static_cast<size_t>(index) * value_stride + column]);
            sum0 += weights0[index] * value;
            sum1 += weights1[index] * value;
            sum2 += weights2[index] * value;
            sum3 += weights3[index] * value;
        }
        output[column] = sum0;
        output[columns + column] = sum1;
        output[2 * columns + column] = sum2;
        output[3 * columns + column] = sum3;
    }
}

static float scalar_f32_scale_max(float* values, float scale, uint32_t count) noexcept
{
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint32_t index = 0; index < count; ++index)
    {
        const float scaled = values[index] * scale;
        values[index] = scaled;
        maximum = std::max(maximum, scaled);
    }
    return maximum;
}

static float scalar_f32_softmax(float* values, uint32_t count, float maximum) noexcept
{
    float sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
    {
        const float exponential = float_approximate_exp(values[index] - maximum);
        values[index] = exponential;
        sum += exponential;
    }
    return sum;
}

static void scalar_float_exp_inplace(float* values, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        values[index] = float_approximate_exp(values[index]);
}

static float scalar_int8_float_dot(const int8_t* left,
                                   const float* right,
                                   uint32_t count) noexcept
{
    float result = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        result += static_cast<float>(left[index]) * right[index];
    return result;
}

static void scalar_float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        output[index] += scale * input[index];
}

static void scalar_float_scale_inplace(float* values, float scale, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        values[index] *= scale;
}

static void scalar_float_rms_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    if (count == 0)
        return;
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        square_sum += values[index] * values[index];
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    for (uint32_t index = 0; index < count; ++index)
        values[index] *= inverse_rms;
}

static void scalar_float_l2_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    if (count == 0)
        return;
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        square_sum += values[index] * values[index];
    const float inverse_norm = 1.0f / std::sqrt(square_sum + epsilon);
    for (uint32_t index = 0; index < count; ++index)
        values[index] *= inverse_norm;
}

static void scalar_float_scale_add(float* output,
                                   float output_scale,
                                   const float* input,
                                   float input_scale,
                                   uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        output[index] = output[index] * output_scale + input[index] * input_scale;
}

static void scalar_float_scale_inplace_and_scaled_add(float* values,
                                                      float value_scale,
                                                      float* output,
                                                      float output_scale,
                                                      uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        values[index] *= value_scale;
        output[index] += output_scale * values[index];
    }
}

static void scalar_float_scaled_add_and_accumulate(float* values,
                                                   const float* input,
                                                   float input_scale,
                                                   float* output,
                                                   float output_scale,
                                                   uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        const float product = input[index] * input_scale;
        values[index] += product;
        output[index] += output_scale * values[index];
    }
}

static void scalar_pv_f32(float* output,
                          const float* values0,
                          size_t value_stride,
                          const float* weights0,
                          uint32_t count0,
                          const float* values1,
                          const float* weights1,
                          uint32_t count1,
                          uint32_t columns) noexcept
{
    for (uint32_t key = 0; key < count0; ++key)
    {
        const float weight = weights0[key];
        const float* row = values0 + static_cast<size_t>(key) * value_stride;
        for (uint32_t column = 0; column < columns; ++column)
            output[column] += weight * row[column];
    }
    for (uint32_t key = 0; key < count1; ++key)
    {
        const float weight = weights1[key];
        const float* row = values1 + static_cast<size_t>(key) * value_stride;
        for (uint32_t column = 0; column < columns; ++column)
            output[column] += weight * row[column];
    }
}

static void scalar_gdn_conv4(float* state,
                             float* values,
                             size_t value_stride,
                             size_t first,
                             size_t row_count,
                             uint32_t begin,
                             uint32_t end,
                             uint32_t columns,
                             const float* taps) noexcept
{
    const float* tap0 = taps;
    const float* tap1 = taps + columns;
    const float* tap2 = taps + static_cast<size_t>(columns) * 2;
    const float* tap3 = taps + static_cast<size_t>(columns) * 3;
    for (size_t token = 0; token < row_count; ++token)
    {
        float* row = values + (first + token) * value_stride;
        for (uint32_t channel = begin; channel < end; ++channel)
        {
            const float x = row[channel];
            const float h1 = state[columns + channel];
            const float h2 = state[static_cast<size_t>(columns) * 2 + channel];
            const float h3 = state[static_cast<size_t>(columns) * 3 + channel];
            row[channel] = h1 * tap0[channel]
                           + h2 * tap1[channel]
                           + h3 * tap2[channel]
                           + x * tap3[channel];
            state[channel] = h1;
            state[columns + channel] = h2;
            state[static_cast<size_t>(columns) * 2 + channel] = h3;
            state[static_cast<size_t>(columns) * 3 + channel] = x;
        }
    }
}

static void scalar_gdn_step(float* state,
                            const float* key,
                            const float* query,
                            const float* value,
                            uint32_t key_dim,
                            uint32_t value_dim,
                            float decay,
                            float beta,
                            float scale,
                            float* memory,
                            float* delta,
                            float* output) noexcept
{
    for (uint32_t column = 0; column < value_dim; ++column)
        memory[column] = 0.0f;
    for (uint32_t row = 0; row < key_dim; ++row)
    {
        const float* state_row = state + static_cast<size_t>(row) * value_dim;
        const float key_value = key[row];
        for (uint32_t column = 0; column < value_dim; ++column)
        {
            const float decayed = state_row[column] * decay;
            memory[column] += decayed * key_value;
        }
    }
    for (uint32_t column = 0; column < value_dim; ++column)
    {
        delta[column] = (value[column] - memory[column]) * beta;
        output[column] = 0.0f;
    }
    for (uint32_t row = 0; row < key_dim; ++row)
    {
        float* state_row = state + static_cast<size_t>(row) * value_dim;
        const float key_value = key[row];
        const float query_value = query[row];
        for (uint32_t column = 0; column < value_dim; ++column)
        {
            const float updated = state_row[column] * decay + key_value * delta[column];
            state_row[column] = updated;
            output[column] += updated * query_value;
        }
    }
    for (uint32_t column = 0; column < value_dim; ++column)
        output[column] *= scale;
}

static void scalar_float_weighted_scale(float* output, const float* input, const float* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        output[index] = input[index] * scale * (weight[index] + weight_offset);
}

static void scalar_bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        const float value = std::bit_cast<float>(static_cast<uint32_t>(weight[index]) << 16);
        output[index] = input[index] * scale * (value + weight_offset);
    }
}

static void scalar_float_rms_norm(float* output,
                                  const float* input,
                                  const float* weight,
                                  float epsilon,
                                  float weight_offset,
                                  uint32_t count) noexcept
{
    if (count == 0)
        return;
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        square_sum += input[index] * input[index];
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    for (uint32_t index = 0; index < count; ++index)
        output[index] = input[index] * inverse_rms * (weight[index] + weight_offset);
}

static void scalar_bfloat16_rms_norm(float* output,
                                     const float* input,
                                     const uint16_t* weight,
                                     float epsilon,
                                     float weight_offset,
                                     uint32_t count) noexcept
{
    if (count == 0)
        return;
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        square_sum += input[index] * input[index];
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    for (uint32_t index = 0; index < count; ++index)
    {
        const float weight_value = std::bit_cast<float>(static_cast<uint32_t>(weight[index]) << 16);
        output[index] = input[index] * inverse_rms * (weight_value + weight_offset);
    }
}

static void scalar_float_rope_inplace(float* values,
                                      const float* cosine,
                                      const float* sine,
                                      uint32_t dimension) noexcept
{
    const uint32_t half_dimension = dimension / 2;
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const float first = values[index];
        const float second = values[half_dimension + index];
        values[index] = first * cosine[index] - second * sine[index];
        values[half_dimension + index] = second * cosine[index] + first * sine[index];
    }
}

static void scalar_float_hc_pre_4(float* output,
                                  const float* input,
                                  float scale0,
                                  float scale1,
                                  float scale2,
                                  float scale3,
                                  uint32_t hidden_size) noexcept
{
    const float* input1 = input + hidden_size;
    const float* input2 = input1 + hidden_size;
    const float* input3 = input2 + hidden_size;
    for (uint32_t index = 0; index < hidden_size; ++index)
        output[index] = input[index] * scale0
                        + input1[index] * scale1
                        + input2[index] * scale2
                        + input3[index] * scale3;
}

static void scalar_float_hc_post_4(float* output,
                                   const float* branch,
                                   const float* residual,
                                   const float* post,
                                   const float* combine,
                                   uint32_t hidden_size) noexcept
{
    for (uint32_t index = 0; index < hidden_size; ++index)
    {
        for (uint32_t output_index = 0; output_index < 4; ++output_index)
        {
            float value = branch[index] * post[output_index];
            for (uint32_t residual_index = 0; residual_index < 4; ++residual_index)
            {
                value += residual[static_cast<size_t>(residual_index) * hidden_size + index]
                         * combine[residual_index * 4 + output_index];
            }
            output[static_cast<size_t>(output_index) * hidden_size + index] = value;
        }
    }
}

static void scalar_float_sigmoid_mul(float* output,
                                     const float* gate,
                                     const float* input,
                                     uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        output[index] = input[index] / (1.0f + float_approximate_exp(-gate[index]));
}

static void scalar_float_silu_mul(float* output, const float* gate, const float* up,
                                  float sigmoid_scale, float up_offset, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        const float gate_value = gate[index];
        const float up_value = up[index];
        const float biased_up = up_offset == 0.0f ? up_value : up_value + up_offset;
        output[index] = gate_value / (1.0f + float_approximate_exp(-sigmoid_scale * gate_value))
                        * biased_up;
    }
}

static void scalar_float_silu_inplace(float* values, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        const float value = values[index];
        values[index] = value / (1.0f + float_approximate_exp(-value));
    }
}

void register_scalar_cpu_ops(CpuKernelOps& ops, CpuBackendCaps& caps) noexcept
{
    ops.float_dot = scalar_float_dot;
    ops.float_argmax = scalar_float_argmax;
    ops.float_gemm_4x4 = scalar_float_gemm_4x4;
    ops.float_gemm_4x8 = scalar_float_gemm_4x8;
    ops.bfloat16_gemm_4x8 = scalar_bfloat16_gemm_4x8;
    ops.f32_to_f16 = scalar_f32_to_f16;
    ops.bf16_to_f16 = scalar_bf16_to_f16;
    ops.f16_gemm_4x8 = scalar_f16_gemm_4x8;
    ops.f16_pv = scalar_f16_pv;
    ops.f16_pv4 = scalar_f16_pv4;
    ops.f32_scale_max = scalar_f32_scale_max;
    ops.f32_softmax = scalar_f32_softmax;
    ops.float_exp_inplace = scalar_float_exp_inplace;
    ops.float_silu_inplace = scalar_float_silu_inplace;
    ops.int8_float_dot = scalar_int8_float_dot;
    ops.float_l2_scale_inplace = scalar_float_l2_scale_inplace;
    ops.float_rms_scale_inplace = scalar_float_rms_scale_inplace;
    ops.float_rms_norm = scalar_float_rms_norm;
    ops.bfloat16_rms_norm = scalar_bfloat16_rms_norm;
    ops.float_rope_inplace = scalar_float_rope_inplace;
    ops.float_scale_inplace = scalar_float_scale_inplace;
    ops.float_scaled_add = scalar_float_scaled_add;
    ops.float_scale_add = scalar_float_scale_add;
    ops.scale_add = scalar_float_scale_inplace_and_scaled_add;
    ops.scaled_acc = scalar_float_scaled_add_and_accumulate;
    ops.gdn_step = scalar_gdn_step;
    ops.gdn_conv4 = scalar_gdn_conv4;
    ops.pv_f32 = scalar_pv_f32;
    ops.float_weighted_scale = scalar_float_weighted_scale;
    ops.bfloat16_weighted_scale = scalar_bfloat16_weighted_scale;
    ops.float_sigmoid_mul = scalar_float_sigmoid_mul;
    ops.float_silu_mul = scalar_float_silu_mul;
    ops.float_hc_pre_4 = scalar_float_hc_pre_4;
    ops.float_hc_post_4 = scalar_float_hc_post_4;

    caps.active_isa_flags = 0;
    caps.arch_name = "scalar";
    caps.has_simd_exp = false;
    caps.has_f16_vector = false;

    register_scalar_bfloat16_ops(ops, caps);
    register_scalar_float8_ops(ops, caps);
    register_scalar_mxfp4_ops(ops, caps);
    register_scalar_qnk_ops(ops, caps);
}

} // namespace moe
} // namespace ncnn
