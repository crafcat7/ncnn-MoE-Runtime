#ifndef NCNN_MOE_KERNELS_CPU_GEMM_TILE_H
#define NCNN_MOE_KERNELS_CPU_GEMM_TILE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

using FloatGemmTileKernel = void (*)(const float* weights,
                                     size_t weight_stride,
                                     const float* input,
                                     size_t input_stride,
                                     uint32_t input_columns,
                                     uint32_t output_count,
                                     uint32_t token_count,
                                     float* output,
                                     size_t output_stride) noexcept;

using Bfloat16GemmTileKernel = void (*)(const uint16_t* weights,
                                        size_t weight_stride,
                                        const float* input,
                                        size_t input_stride,
                                        uint32_t input_columns,
                                        uint32_t output_count,
                                        uint32_t token_count,
                                        float* output,
                                        size_t output_stride) noexcept;

inline void run_tiled_float_gemm(uint32_t tile_tokens,
                                 uint32_t tile_outputs,
                                 FloatGemmTileKernel kernel,
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
    if (token_count == 0 || output_count == 0)
        return;
    for (uint32_t token = 0; token < token_count; token += tile_tokens)
    {
        const uint32_t rows = std::min(tile_tokens, token_count - token);
        const float* tile_input = input + static_cast<size_t>(token) * input_stride;
        float* tile_output = output + static_cast<size_t>(token) * output_stride;
        for (uint32_t out = 0; out < output_count; out += tile_outputs)
        {
            const uint32_t cols = std::min(tile_outputs, output_count - out);
            kernel(weights + static_cast<size_t>(out) * weight_stride,
                   weight_stride,
                   tile_input,
                   input_stride,
                   input_columns,
                   cols,
                   rows,
                   tile_output + out,
                   output_stride);
        }
    }
}

inline void run_tiled_bfloat16_gemm(uint32_t tile_tokens,
                                    uint32_t tile_outputs,
                                    Bfloat16GemmTileKernel kernel,
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
    if (token_count == 0 || output_count == 0)
        return;
    for (uint32_t token = 0; token < token_count; token += tile_tokens)
    {
        const uint32_t rows = std::min(tile_tokens, token_count - token);
        const float* tile_input = input + static_cast<size_t>(token) * input_stride;
        float* tile_output = output + static_cast<size_t>(token) * output_stride;
        for (uint32_t out = 0; out < output_count; out += tile_outputs)
        {
            const uint32_t cols = std::min(tile_outputs, output_count - out);
            kernel(weights + static_cast<size_t>(out) * weight_stride,
                   weight_stride,
                   tile_input,
                   input_stride,
                   input_columns,
                   cols,
                   rows,
                   tile_output + out,
                   output_stride);
        }
    }
}

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_KERNELS_CPU_GEMM_TILE_H
