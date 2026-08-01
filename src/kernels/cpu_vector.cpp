#include "cpu_vector.h"

#include "engine/cpu_features.h"
#include "ncnn/moe/runtime.h"

#if defined(NCNN_MOE_MSVC_X86_SIMD)
#include "cpu_vector_msvc.h"
#endif

namespace ncnn {
namespace moe {

using FloatDotFunction = float (*)(const float*, const float*, uint32_t) noexcept;
using FloatScaledAddFunction = void (*)(float*, const float*, float, uint32_t) noexcept;

static float scalar_float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    float result = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        result += left[index] * right[index];
    return result;
}

static void scalar_float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        output[index] += scale * input[index];
}

static FloatDotFunction select_float_dot() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = detect_cpu_isa_capabilities().flags;
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_dot;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_dot;
#endif
    return scalar_float_dot;
}

static FloatScaledAddFunction select_float_scaled_add() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    const uint64_t isa = detect_cpu_isa_capabilities().flags;
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_float_scaled_add;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_scaled_add;
#endif
    return scalar_float_scaled_add;
}

float float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    static const FloatDotFunction function = select_float_dot();
    return function(left, right, count);
}

void float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept
{
    static const FloatScaledAddFunction function = select_float_scaled_add();
    function(output, input, scale, count);
}

} // namespace moe
} // namespace ncnn
