#ifndef NCNN_MOE_VECTOR_H
#define NCNN_MOE_VECTOR_H

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

float float_dot(const float* left, const float* right, uint32_t count) noexcept;
size_t float_argmax(const float* values, size_t count) noexcept;
void float_gemm_4x4(const float* weights,
                    size_t weight_stride,
                    const float* input,
                    size_t input_stride,
                    uint32_t input_columns,
                    uint32_t output_count,
                    uint32_t token_count,
                    float* output,
                    size_t output_stride) noexcept;
// Computes up to four input rows by up to eight output rows.  The caller
// keeps the tile within those bounds and uses this for small-batch GEMM
// dispatch so the input tile and each weight vector are reused across rows.
void float_gemm_4x8(const float* weights,
                    size_t weight_stride,
                    const float* input,
                    size_t input_stride,
                    uint32_t input_columns,
                    uint32_t output_count,
                    uint32_t token_count,
                    float* output,
                    size_t output_stride) noexcept;
void bfloat16_gemm_4x8(const uint16_t* weights,
                       size_t weight_stride,
                       const float* input,
                       size_t input_stride,
                       uint32_t input_columns,
                       uint32_t output_count,
                       uint32_t token_count,
                       float* output,
                       size_t output_stride) noexcept;
void float_exp_inplace(float* values, uint32_t count) noexcept;
[[nodiscard]] bool float_exp_simd_available() noexcept;
void float_silu_inplace(float* values, uint32_t count) noexcept;
float int8_float_dot(const int8_t* left, const float* right, uint32_t count) noexcept;
void float_l2_scale_inplace(float* values, float epsilon, uint32_t count) noexcept;
void float_rms_scale_inplace(float* values, float epsilon, uint32_t count) noexcept;
void float_rms_norm(float* output, const float* input, const float* weight,
                    float epsilon, float weight_offset, uint32_t count) noexcept;
void bfloat16_rms_norm(float* output, const float* input, const uint16_t* weight,
                       float epsilon, float weight_offset, uint32_t count) noexcept;
void float_rope_inplace(float* values, const float* cosine, const float* sine,
                        uint32_t dimension) noexcept;
void float_scale_inplace(float* values, float scale, uint32_t count) noexcept;
void float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept;
void float_scale_add(float* output, float output_scale, const float* input, float input_scale, uint32_t count) noexcept;
void scale_add(float* values,
               float value_scale,
               float* output,
               float output_scale,
               uint32_t count) noexcept;
void scaled_acc(float* values,
                const float* input,
                float input_scale,
                float* output,
                float output_scale,
                uint32_t count) noexcept;
// One token of the gated delta rule for a single head.  The state is
// [key_dim][value_dim]; the key/query vectors have key_dim entries and the
// value vector has value_dim entries.  memory/delta/output are caller scratch
// of value_dim floats.
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
              float* output) noexcept;

// Causal depthwise convolution for one gated delta layer.  The state layout is
// [tap][channel] and taps is the same layout in FP32, so both the state and the
// filter are contiguous across channels.
void gdn_conv4(float* state,
               float* values,
               size_t value_stride,
               size_t first,
               size_t row_count,
               uint32_t begin,
               uint32_t end,
               uint32_t columns,
               const float* taps) noexcept;

void pv_f32(float* output,
            const float* values0,
            size_t value_stride,
            const float* weights0,
            uint32_t count0,
            const float* values1,
            const float* weights1,
            uint32_t count1,
            uint32_t columns) noexcept;
void float_weighted_scale(float* output, const float* input, const float* weight,
                          float scale, float weight_offset, uint32_t count) noexcept;
void bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight,
                             float scale, float weight_offset, uint32_t count) noexcept;
void float_sigmoid_mul(float* output, const float* gate, const float* input, uint32_t count) noexcept;
void float_silu_mul(float* output, const float* gate, const float* up,
                    float sigmoid_scale, float up_offset, uint32_t count) noexcept;
void float_hc_pre_4(float* output, const float* input,
                    float scale0, float scale1, float scale2, float scale3,
                    uint32_t hidden_size) noexcept;
void float_hc_post_4(float* output, const float* branch, const float* residual,
                     const float* post, const float* combine,
                     uint32_t hidden_size) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_VECTOR_H
