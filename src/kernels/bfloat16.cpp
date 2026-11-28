#include "kernels/bfloat16.h"

#include "kernels/cpu/cpu_ops.h"
#include "ncnn/moe/option.h"

#include <limits>
#include <vector>

namespace ncnn {
namespace moe {

const char* bfloat16_dot_kernel_name() noexcept
{
    return cpu_backend_caps().bfloat16_dot_name;
}

float bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    return cpu_ops().bfloat16_dot(weights, input, count);
}

void float_to_bfloat16_array(uint16_t* output,
                             const float* input,
                             uint32_t count) noexcept
{
    cpu_ops().float_to_bfloat16_array(output, input, count);
}

float bfloat16_pair_dot(const uint16_t* left,
                        const uint16_t* right,
                        uint32_t count) noexcept
{
    return cpu_ops().bfloat16_pair_dot(left, right, count);
}

void bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
{
    cpu_ops().bfloat16_scaled_add(output, input, scale, count);
}

void pv_bf16(float* output,
             const uint16_t* values0,
             size_t value_stride,
             const float* weights0,
             uint32_t count0,
             const uint16_t* values1,
             const float* weights1,
             uint32_t count1,
             uint32_t columns) noexcept
{
    cpu_ops().pv_bf16(output, values0, value_stride, weights0, count0, values1, weights1, count1, columns);
}

const char* bfloat16_batched_linear_kernel_name(uint64_t optimization_flags) noexcept
{
    if (!has_flag(optimization_flags, OptimizationCpuBfloat16Batched)
        || !cpu_backend_caps().bf16_fast_dp)
    {
        return "unavailable";
    }
    return "x86-avx512-bf16-dpbf16-1x4+4x4";
}

bool forward_linear_bf16(const uint16_t* weights,
                         const float* input,
                         size_t input_stride,
                         size_t token_count,
                         uint32_t output_columns,
                         uint32_t input_columns,
                         float* output,
                         size_t output_stride,
                         int thread_count,
                         uint64_t optimization_flags)
{
    if (!weights || !input || !output || token_count == 0
        || output_columns < 4 || output_columns % 4 != 0
        || input_columns == 0 || input_columns % 32 != 0
        || input_stride < input_columns || output_stride < output_columns)
    {
        return false;
    }
    if (token_count > std::numeric_limits<size_t>::max() - 3)
        return false;
    const size_t padded_token_count = (token_count + 3) & ~size_t{3};
    if (padded_token_count > std::numeric_limits<size_t>::max() / input_columns)
        return false;

    if (!has_flag(optimization_flags, OptimizationCpuBfloat16Batched)
        || !cpu_backend_caps().bf16_fast_dp)
    {
        return false;
    }
    if (token_count > std::numeric_limits<uint64_t>::max() / output_columns / input_columns)
        return false;
    const uint64_t operation_count = static_cast<uint64_t>(token_count) * output_columns * input_columns;
    if (output_columns < 128 || operation_count < 1024 * 1024)
        return false;

    const CpuKernelOps& ops = cpu_ops();
    const int team_size = thread_count > 1 ? thread_count : 1;
    if (token_count == 1 && ops.bfloat16_single_token_linear != nullptr)
    {
        ops.bfloat16_single_token_linear(weights, input, output_columns, input_columns, output, team_size);
        return true;
    }
    if (ops.bfloat16_batched_linear != nullptr)
    {
        static thread_local std::vector<uint16_t> packed_input;
        ops.bfloat16_batched_linear(weights, input, input_stride, token_count, output_columns,
                                    input_columns, output, output_stride, team_size, packed_input);
        return true;
    }
    return false;
}

} // namespace moe
} // namespace ncnn
