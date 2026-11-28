#include "vector.h"

#include "kernels/cpu/cpu_ops.h"

namespace ncnn {
namespace moe {

float float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    return cpu_ops().float_dot(left, right, count);
}

size_t float_argmax(const float* values, size_t count) noexcept
{
    return cpu_ops().float_argmax(values, count);
}

void float_gemm_4x4(const float* weights,
                    size_t weight_stride,
                    const float* input,
                    size_t input_stride,
                    uint32_t input_columns,
                    uint32_t output_count,
                    uint32_t token_count,
                    float* output,
                    size_t output_stride) noexcept
{
    cpu_ops().float_gemm_4x4(weights, weight_stride, input, input_stride,
                             input_columns, output_count, token_count, output, output_stride);
}

void float_gemm_4x8(const float* weights,
                    size_t weight_stride,
                    const float* input,
                    size_t input_stride,
                    uint32_t input_columns,
                    uint32_t output_count,
                    uint32_t token_count,
                    float* output,
                    size_t output_stride) noexcept
{
    cpu_ops().float_gemm_4x8(weights, weight_stride, input, input_stride,
                             input_columns, output_count, token_count, output, output_stride);
}

void bfloat16_gemm_4x8(const uint16_t* weights,
                       size_t weight_stride,
                       const float* input,
                       size_t input_stride,
                       uint32_t input_columns,
                       uint32_t output_count,
                       uint32_t token_count,
                       float* output,
                       size_t output_stride) noexcept
{
    cpu_ops().bfloat16_gemm_4x8(weights, weight_stride, input, input_stride,
                                input_columns, output_count, token_count, output, output_stride);
}

void f32_to_f16(uint16_t* output, const float* input, size_t count) noexcept
{
    cpu_ops().f32_to_f16(output, input, count);
}

void bf16_to_f16(uint16_t* output, const uint16_t* input, size_t count) noexcept
{
    cpu_ops().bf16_to_f16(output, input, count);
}

void f16_gemm_4x8(const uint16_t* weights,
                  size_t weight_stride,
                  const uint16_t* input,
                  size_t input_stride,
                  uint32_t input_columns,
                  uint32_t output_count,
                  uint32_t token_count,
                  float* output,
                  size_t output_stride) noexcept
{
    cpu_ops().f16_gemm_4x8(weights, weight_stride, input, input_stride,
                           input_columns, output_count, token_count, output, output_stride);
}

void f16_pv(float* output,
            const uint16_t* values,
            size_t value_stride,
            const uint16_t* weights,
            uint32_t count,
            uint32_t columns) noexcept
{
    cpu_ops().f16_pv(output, values, value_stride, weights, count, columns);
}

void f16_pv4(float* output,
             const uint16_t* values,
             size_t value_stride,
             const float* weights0,
             const float* weights1,
             const float* weights2,
             const float* weights3,
             uint32_t count,
             uint32_t columns) noexcept
{
    cpu_ops().f16_pv4(output, values, value_stride, weights0, weights1, weights2, weights3, count, columns);
}

float f32_scale_max(float* values, float scale, uint32_t count) noexcept
{
    return cpu_ops().f32_scale_max(values, scale, count);
}

float f32_softmax(float* values, uint32_t count, float maximum) noexcept
{
    return cpu_ops().f32_softmax(values, count, maximum);
}

void float_exp_inplace(float* values, uint32_t count) noexcept
{
    cpu_ops().float_exp_inplace(values, count);
}

bool float_exp_simd_available() noexcept
{
    return cpu_backend_caps().has_simd_exp;
}

float int8_float_dot(const int8_t* left, const float* right, uint32_t count) noexcept
{
    return cpu_ops().int8_float_dot(left, right, count);
}

void float_l2_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    cpu_ops().float_l2_scale_inplace(values, epsilon, count);
}

void float_rms_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    cpu_ops().float_rms_scale_inplace(values, epsilon, count);
}

void float_rms_norm(float* output, const float* input, const float* weight,
                    float epsilon, float weight_offset, uint32_t count) noexcept
{
    cpu_ops().float_rms_norm(output, input, weight, epsilon, weight_offset, count);
}

void bfloat16_rms_norm(float* output, const float* input, const uint16_t* weight,
                       float epsilon, float weight_offset, uint32_t count) noexcept
{
    cpu_ops().bfloat16_rms_norm(output, input, weight, epsilon, weight_offset, count);
}

void float_rope_inplace(float* values, const float* cosine, const float* sine,
                        uint32_t dimension) noexcept
{
    cpu_ops().float_rope_inplace(values, cosine, sine, dimension);
}

void float_scale_inplace(float* values, float scale, uint32_t count) noexcept
{
    cpu_ops().float_scale_inplace(values, scale, count);
}

void float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept
{
    cpu_ops().float_scaled_add(output, input, scale, count);
}

void float_scale_add(float* output,
                     float output_scale,
                     const float* input,
                     float input_scale,
                     uint32_t count) noexcept
{
    cpu_ops().float_scale_add(output, output_scale, input, input_scale, count);
}

void scale_add(float* values,
               float value_scale,
               float* output,
               float output_scale,
               uint32_t count) noexcept
{
    cpu_ops().scale_add(values, value_scale, output, output_scale, count);
}

void scaled_acc(float* values,
                const float* input,
                float input_scale,
                float* output,
                float output_scale,
                uint32_t count) noexcept
{
    cpu_ops().scaled_acc(values, input, input_scale, output, output_scale, count);
}

void gdn_step(float* state,
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
    cpu_ops().gdn_step(state, key, query, value, key_dim, value_dim, decay, beta, scale, memory, delta, output);
}

void gdn_conv4(float* state,
               float* values,
               size_t value_stride,
               size_t first,
               size_t row_count,
               uint32_t begin,
               uint32_t end,
               uint32_t columns,
               const float* taps) noexcept
{
    cpu_ops().gdn_conv4(state, values, value_stride, first, row_count, begin, end, columns, taps);
}

void pv_f32(float* output,
            const float* values0,
            size_t value_stride,
            const float* weights0,
            uint32_t count0,
            const float* values1,
            const float* weights1,
            uint32_t count1,
            uint32_t columns) noexcept
{
    cpu_ops().pv_f32(output, values0, value_stride, weights0, count0, values1, weights1, count1, columns);
}

void float_weighted_scale(float* output, const float* input, const float* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    cpu_ops().float_weighted_scale(output, input, weight, scale, weight_offset, count);
}

void bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    cpu_ops().bfloat16_weighted_scale(output, input, weight, scale, weight_offset, count);
}

void float_sigmoid_mul(float* output,
                       const float* gate,
                       const float* input,
                       uint32_t count) noexcept
{
    cpu_ops().float_sigmoid_mul(output, gate, input, count);
}

void float_silu_mul(float* output, const float* gate, const float* up,
                    float sigmoid_scale, float up_offset, uint32_t count) noexcept
{
    cpu_ops().float_silu_mul(output, gate, up, sigmoid_scale, up_offset, count);
}

void float_silu_inplace(float* values, uint32_t count) noexcept
{
    cpu_ops().float_silu_inplace(values, count);
}

void float_hc_pre_4(float* output, const float* input,
                    float scale0, float scale1, float scale2, float scale3,
                    uint32_t hidden_size) noexcept
{
    cpu_ops().float_hc_pre_4(output, input, scale0, scale1, scale2, scale3, hidden_size);
}

void float_hc_post_4(float* output, const float* branch, const float* residual,
                     const float* post, const float* combine,
                     uint32_t hidden_size) noexcept
{
    cpu_ops().float_hc_post_4(output, branch, residual, post, combine, hidden_size);
}

} // namespace moe
} // namespace ncnn
