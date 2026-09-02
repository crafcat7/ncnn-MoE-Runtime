#include "cpu_vector.h"

#include "cpu_fast_math.h"
#include "engine/cpu_features.h"
#include "ncnn/moe/runtime.h"

#include <bit>
#include <cmath>

#if defined(NCNN_MOE_MSVC_X86_SIMD)
#include "cpu_vector_msvc.h"
#endif

namespace ncnn {
namespace moe {

using FloatDotFunction = float (*)(const float*, const float*, uint32_t) noexcept;
using FloatExpInplaceFunction = void (*)(float*, uint32_t) noexcept;
using Int8FloatDotFunction = float (*)(const int8_t*, const float*, uint32_t) noexcept;
using FloatScaleFunction = void (*)(float*, float, uint32_t) noexcept;
using FloatScaledAddFunction = void (*)(float*, const float*, float, uint32_t) noexcept;
using FloatScaleAddFunction = void (*)(float*, float, const float*, float, uint32_t) noexcept;
using FloatScaleInplaceAndScaledAddFunction = void (*)(float*, float, float*, float, uint32_t) noexcept;
using FloatScaleInplaceAndScaledAddAndAccumulateFunction = void (*)(float*, float, const float*, float, float*, float, uint32_t) noexcept;
using FloatWeightedScaleFunction = void (*)(float*, const float*, const float*, float, float, uint32_t) noexcept;
using Bfloat16WeightedScaleFunction = void (*)(float*, const float*, const uint16_t*, float, float, uint32_t) noexcept;
using FloatRmsScaleFunction = void (*)(float*, float, uint32_t) noexcept;
using FloatL2ScaleFunction = void (*)(float*, float, uint32_t) noexcept;
using FloatRmsNormFunction = void (*)(float*, const float*, const float*, float, float, uint32_t) noexcept;
using Bfloat16RmsNormFunction = void (*)(float*, const float*, const uint16_t*, float, float, uint32_t) noexcept;
using FloatRopeFunction = void (*)(float*, const float*, const float*, uint32_t) noexcept;
using FloatHcPre4Function = void (*)(float*, const float*, float, float, float, float, uint32_t) noexcept;
using FloatHcPost4Function = void (*)(float*, const float*, const float*, const float*, const float*, uint32_t) noexcept;

static float scalar_float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    float result = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        result += left[index] * right[index];
    return result;
}

static void scalar_float_gemm_4x4(
    const float* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept
{
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

static void scalar_float_gemm_4x8(
    const float* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept
{
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

static void scalar_bfloat16_gemm_4x8(
    const uint16_t* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept
{
    float accumulators[4][8] = {};
    for (uint32_t column = 0; column < input_columns; ++column)
    {
        float input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
            input_values[token] = input[static_cast<size_t>(token) * input_stride + column];
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const uint32_t bits = static_cast<uint32_t>(
                                      weights[static_cast<size_t>(output_index) * weight_stride + column])
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

static float scalar_int8_float_dot(
    const int8_t* left,
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

static void scalar_float_scale_add(
    float* output,
    float output_scale,
    const float* input,
    float input_scale,
    uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        output[index] = output[index] * output_scale + input[index] * input_scale;
}

static void scalar_float_scale_inplace_and_scaled_add(
    float* values,
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

static void scalar_float_scale_inplace_and_scaled_add_and_accumulate(
    float* values,
    float value_scale,
    const float* input,
    float input_scale,
    float* output,
    float output_scale,
    uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        values[index] = values[index] * value_scale
                        + input[index] * input_scale;
        output[index] += output_scale * values[index];
    }
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

static void scalar_float_rms_norm(
    float* output,
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

static void scalar_bfloat16_rms_norm(
    float* output,
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

static void scalar_float_rope_inplace(
    float* values,
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

static void scalar_float_hc_pre_4(
    float* output,
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

static void scalar_float_hc_post_4(
    float* output,
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

static void scalar_float_sigmoid_mul(
    float* output,
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
        output[index] = gate_value / (1.0f + float_approximate_exp(-sigmoid_scale * gate_value))
                        * (up[index] + up_offset);
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

static FloatDotFunction select_float_dot() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_dot;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_dot;
#endif
    return scalar_float_dot;
}

static FloatExpInplaceFunction select_float_exp_inplace() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_exp_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_exp_inplace;
#endif
    return nullptr;
}

static Int8FloatDotFunction select_int8_float_dot() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_int8_float_dot;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_int8_float_dot;
#endif
    return scalar_int8_float_dot;
}

static FloatScaledAddFunction select_float_scaled_add() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_scaled_add;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_scaled_add;
#endif
    return scalar_float_scaled_add;
}

static FloatScaleFunction select_float_scale() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_scale_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_scale_inplace;
#endif
    return scalar_float_scale_inplace;
}

static FloatScaleAddFunction select_float_scale_add() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_scale_add;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_scale_add;
#endif
    return scalar_float_scale_add;
}

static FloatScaleInplaceAndScaledAddFunction select_float_scale_inplace_and_scaled_add() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_scale_inplace_and_scaled_add;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_scale_inplace_and_scaled_add;
#endif
    return scalar_float_scale_inplace_and_scaled_add;
}

static FloatScaleInplaceAndScaledAddAndAccumulateFunction
select_float_scale_inplace_and_scaled_add_and_accumulate() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_scale_inplace_and_scaled_add_and_accumulate;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_scale_inplace_and_scaled_add_and_accumulate;
#endif
    return scalar_float_scale_inplace_and_scaled_add_and_accumulate;
}

static FloatWeightedScaleFunction select_float_weighted_scale() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_weighted_scale;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_weighted_scale;
#endif
    return scalar_float_weighted_scale;
}

static Bfloat16WeightedScaleFunction select_bfloat16_weighted_scale() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_bfloat16_weighted_scale;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_bfloat16_weighted_scale;
#endif
    return scalar_bfloat16_weighted_scale;
}

static FloatRmsScaleFunction select_float_rms_scale() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_rms_scale_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_rms_scale_inplace;
#endif
    return scalar_float_rms_scale_inplace;
}

static FloatL2ScaleFunction select_float_l2_scale() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_l2_scale_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_l2_scale_inplace;
#endif
    return scalar_float_l2_scale_inplace;
}

static FloatRmsNormFunction select_float_rms_norm() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_rms_norm;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_rms_norm;
#endif
    return scalar_float_rms_norm;
}

static Bfloat16RmsNormFunction select_bfloat16_rms_norm() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_bfloat16_rms_norm;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_bfloat16_rms_norm;
#endif
    return scalar_bfloat16_rms_norm;
}

static FloatRopeFunction select_float_rope() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_rope_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_rope_inplace;
#endif
    return scalar_float_rope_inplace;
}

static FloatHcPre4Function select_float_hc_pre_4() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_hc_pre_4;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_hc_pre_4;
#endif
    return scalar_float_hc_pre_4;
}

static FloatHcPost4Function select_float_hc_post_4() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_hc_post_4;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_hc_post_4;
#endif
    return scalar_float_hc_post_4;
}

using FloatSigmoidMulFunction = void (*)(float*, const float*, const float*, uint32_t) noexcept;
using FloatSiluMulFunction = void (*)(float*, const float*, const float*, float, float, uint32_t) noexcept;
using FloatSiluInplaceFunction = void (*)(float*, uint32_t) noexcept;

static FloatSigmoidMulFunction select_float_sigmoid_mul() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_sigmoid_mul;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_sigmoid_mul;
#endif
    return scalar_float_sigmoid_mul;
}

static FloatSiluMulFunction select_float_silu_mul() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_silu_mul;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_silu_mul;
#endif
    return scalar_float_silu_mul;
}

static FloatSiluInplaceFunction select_float_silu_inplace() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_silu_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_silu_inplace;
#endif
    return scalar_float_silu_inplace;
}

float float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    static const FloatDotFunction function = select_float_dot();
    return function(left, right, count);
}

void float_gemm_4x4(
    const float* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
    {
        msvc_avx512_float_gemm_4x4(
            weights,
            weight_stride,
            input,
            input_stride,
            input_columns,
            output_count,
            token_count,
            output,
            output_stride);
        return;
    }
    if ((isa & CpuIsaX86Avx2Fma) != 0)
    {
        msvc_avx2_float_gemm_4x4(
            weights,
            weight_stride,
            input,
            input_stride,
            input_columns,
            output_count,
            token_count,
            output,
            output_stride);
        return;
    }
#endif
    scalar_float_gemm_4x4(
        weights,
        weight_stride,
        input,
        input_stride,
        input_columns,
        output_count,
        token_count,
        output,
        output_stride);
}

void float_gemm_4x8(
    const float* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
    {
        msvc_avx512_float_gemm_4x8(
            weights, weight_stride, input, input_stride, input_columns,
            output_count, token_count, output, output_stride);
        return;
    }
    if ((isa & CpuIsaX86Avx2Fma) != 0)
    {
        msvc_avx2_float_gemm_4x8(
            weights, weight_stride, input, input_stride, input_columns,
            output_count, token_count, output, output_stride);
        return;
    }
#endif
    scalar_float_gemm_4x8(
        weights, weight_stride, input, input_stride, input_columns,
        output_count, token_count, output, output_stride);
}

void bfloat16_gemm_4x8(
    const uint16_t* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
    {
        msvc_avx512_bfloat16_gemm_4x8(
            weights, weight_stride, input, input_stride, input_columns,
            output_count, token_count, output, output_stride);
        return;
    }
    if ((isa & CpuIsaX86Avx2Fma) != 0)
    {
        msvc_avx2_bfloat16_gemm_4x8(
            weights, weight_stride, input, input_stride, input_columns,
            output_count, token_count, output, output_stride);
        return;
    }
#endif
    scalar_bfloat16_gemm_4x8(
        weights, weight_stride, input, input_stride, input_columns,
        output_count, token_count, output, output_stride);
}

void float_exp_inplace(float* values, uint32_t count) noexcept
{
    static const FloatExpInplaceFunction function = select_float_exp_inplace();
    if (function)
    {
        function(values, count);
        return;
    }
    for (uint32_t index = 0; index < count; ++index)
        values[index] = float_approximate_exp(values[index]);
}

bool float_exp_simd_available() noexcept
{
    static const FloatExpInplaceFunction function = select_float_exp_inplace();
    return function != nullptr;
}

float int8_float_dot(const int8_t* left, const float* right, uint32_t count) noexcept
{
    static const Int8FloatDotFunction function = select_int8_float_dot();
    return function(left, right, count);
}

void float_l2_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    static const FloatL2ScaleFunction function = select_float_l2_scale();
    function(values, epsilon, count);
}

void float_rms_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    static const FloatRmsScaleFunction function = select_float_rms_scale();
    function(values, epsilon, count);
}

void float_rms_norm(float* output, const float* input, const float* weight,
                    float epsilon, float weight_offset, uint32_t count) noexcept
{
    static const FloatRmsNormFunction function = select_float_rms_norm();
    function(output, input, weight, epsilon, weight_offset, count);
}

void bfloat16_rms_norm(float* output, const float* input, const uint16_t* weight,
                       float epsilon, float weight_offset, uint32_t count) noexcept
{
    static const Bfloat16RmsNormFunction function = select_bfloat16_rms_norm();
    function(output, input, weight, epsilon, weight_offset, count);
}

void float_rope_inplace(float* values, const float* cosine, const float* sine,
                        uint32_t dimension) noexcept
{
    static const FloatRopeFunction function = select_float_rope();
    function(values, cosine, sine, dimension);
}

void float_scale_inplace(float* values, float scale, uint32_t count) noexcept
{
    static const FloatScaleFunction function = select_float_scale();
    function(values, scale, count);
}

void float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept
{
    static const FloatScaledAddFunction function = select_float_scaled_add();
    function(output, input, scale, count);
}

void float_scale_add(
    float* output,
    float output_scale,
    const float* input,
    float input_scale,
    uint32_t count) noexcept
{
    static const FloatScaleAddFunction function = select_float_scale_add();
    function(output, output_scale, input, input_scale, count);
}

void float_scale_inplace_and_scaled_add(
    float* values,
    float value_scale,
    float* output,
    float output_scale,
    uint32_t count) noexcept
{
    static const FloatScaleInplaceAndScaledAddFunction function = select_float_scale_inplace_and_scaled_add();
    function(values, value_scale, output, output_scale, count);
}

void float_scale_inplace_and_scaled_add_and_accumulate(
    float* values,
    float value_scale,
    const float* input,
    float input_scale,
    float* output,
    float output_scale,
    uint32_t count) noexcept
{
    static const FloatScaleInplaceAndScaledAddAndAccumulateFunction function =
        select_float_scale_inplace_and_scaled_add_and_accumulate();
    function(values, value_scale, input, input_scale, output, output_scale, count);
}

void float_weighted_scale(float* output, const float* input, const float* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    static const FloatWeightedScaleFunction function = select_float_weighted_scale();
    function(output, input, weight, scale, weight_offset, count);
}

void bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    static const Bfloat16WeightedScaleFunction function = select_bfloat16_weighted_scale();
    function(output, input, weight, scale, weight_offset, count);
}

void float_sigmoid_mul(
    float* output,
    const float* gate,
    const float* input,
    uint32_t count) noexcept
{
    static const FloatSigmoidMulFunction function = select_float_sigmoid_mul();
    function(output, gate, input, count);
}

void float_silu_mul(float* output, const float* gate, const float* up,
                    float sigmoid_scale, float up_offset, uint32_t count) noexcept
{
    static const FloatSiluMulFunction function = select_float_silu_mul();
    function(output, gate, up, sigmoid_scale, up_offset, count);
}

void float_silu_inplace(float* values, uint32_t count) noexcept
{
    static const FloatSiluInplaceFunction function = select_float_silu_inplace();
    function(values, count);
}

void float_hc_pre_4(float* output, const float* input,
                    float scale0, float scale1, float scale2, float scale3,
                    uint32_t hidden_size) noexcept
{
    static const FloatHcPre4Function function = select_float_hc_pre_4();
    function(output, input, scale0, scale1, scale2, scale3, hidden_size);
}

void float_hc_post_4(float* output, const float* branch, const float* residual,
                     const float* post, const float* combine,
                     uint32_t hidden_size) noexcept
{
    static const FloatHcPost4Function function = select_float_hc_post_4();
    function(output, branch, residual, post, combine, hidden_size);
}

} // namespace moe
} // namespace ncnn
