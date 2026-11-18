#ifndef NCNN_MOE_ACTIVATION_H
#define NCNN_MOE_ACTIVATION_H

#include <cmath>

namespace ncnn {
namespace moe {

// Tanh-approximated GELU, matching PyTorch's gelu_pytorch_tanh:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// The two coefficients are part of the published approximation, not tunables.
constexpr float gelu_tanh_sqrt_two_over_pi = 0.7978845608028654f;
constexpr float gelu_tanh_cubic = 0.044715f;

[[nodiscard]] inline float gelu_tanh(float value) noexcept
{
    const float inner = value + gelu_tanh_cubic * value * value * value;
    return 0.5f * value * (1.0f + std::tanh(gelu_tanh_sqrt_two_over_pi * inner));
}

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_ACTIVATION_H
