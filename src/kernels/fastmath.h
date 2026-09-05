#ifndef NCNN_MOE_FASTMATH_H
#define NCNN_MOE_FASTMATH_H

#include <bit>
#include <cmath>
#include <cstdint>

namespace ncnn {
namespace moe {

// ncnn's scalar expf range reduction, adapted for the runtime's float32 CPU
// kernels.  The inputs in the softmax and sigmoid paths are finite in normal
// operation; the explicit NaN/overflow guards keep the helper well-defined for
// malformed tensors and for the stable softmax boundary values.
inline float float_approximate_exp(float value) noexcept
{
    if (value != value)
        return value;
    if (value >= 104.0f)
        return INFINITY;
    if (value <= -104.0f)
        return 0.0f;
    if (value < 0.0f)
        return 1.0f / float_approximate_exp(-value);

    // exp(value) = 2^i * exp(f), f in approximately [-log(2)/2, log(2)/2].
    // The magic rounding constant is the same technique used by ncnn's x86
    // exp kernels and avoids a scalar libm round call.
    constexpr float inverse_log_two = 0x1.715476p+0f;
    constexpr float rounding = 0x1.8p23f;
    const float exponent = std::fma(value, inverse_log_two, rounding) - rounding;
    const float remainder = std::fma(
        exponent,
        -0x1.62e4p-1f,
        std::fma(exponent, -0x1.7f7d1cp-20f, value));

    // Degree-7 minimax-like polynomial from ncnn/simplemath.cpp.
    float polynomial = 1.37805939e-3f;
    polynomial = std::fma(polynomial, remainder, 8.37312452e-3f);
    polynomial = std::fma(polynomial, remainder, 4.16695364e-2f);
    polynomial = std::fma(polynomial, remainder, 1.66664720e-1f);
    polynomial = std::fma(polynomial, remainder, 4.99999851e-1f);
    polynomial = std::fma(polynomial, remainder, 1.0f);
    polynomial = std::fma(polynomial, remainder, 1.0f);

    const int exponent_integer = static_cast<int>(exponent);
    const uint32_t exponent_bits = static_cast<uint32_t>(exponent_integer) << 23;
    // Split the scale for negative exponents so subnormal results do not
    // require constructing an invalid normal exponent in one operation.
    const uint32_t underflow_bias = exponent_integer > 0 ? 0u : 0x83000000u;
    const float scale_high = std::bit_cast<float>(0x7f000000u + underflow_bias);
    const float scale_low = std::bit_cast<float>(exponent_bits - underflow_bias);
    return (polynomial * scale_high) * scale_low;
}

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_FASTMATH_H
