#include "cpu_float8.h"

#include "engine/cpu_features.h"
#include "ncnn/moe/runtime.h"

#if defined(NCNN_MOE_MSVC_X86_SIMD)
#include "cpu_float8_msvc.h"
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace ncnn {
namespace moe {

static std::array<float, 256> make_float8_table()
{
    std::array<float, 256> values{};
    for (uint32_t byte = 0; byte < values.size(); ++byte)
    {
        const bool negative = (byte & 0x80u) != 0;
        const uint32_t exponent = byte >> 3 & 0x0fu;
        const uint32_t mantissa = byte & 0x07u;
        float value = 0.0f;
        if (exponent == 0)
        {
            value = std::ldexp(static_cast<float>(mantissa), -9);
        }
        else if (exponent == 15 && mantissa == 7)
        {
            value = std::numeric_limits<float>::quiet_NaN();
        }
        else
        {
            value = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, static_cast<int>(exponent) - 7);
        }
        values[byte] = negative ? -value : value;
    }
    return values;
}

static const std::array<float, 256>& float8_table()
{
    static const std::array<float, 256> table = make_float8_table();
    return table;
}

float float8_e4m3_to_float(uint8_t value) noexcept
{
    return float8_table()[value];
}

uint8_t float_to_float8_e4m3(float value) noexcept
{
    if (std::isnan(value))
        return UINT8_C(0x7f);
    const uint8_t sign = std::signbit(value) ? UINT8_C(0x80) : UINT8_C(0);
    const float magnitude = std::min(std::fabs(value), 448.0f);
    if (magnitude < std::ldexp(1.0f, -6))
    {
        const uint32_t mantissa = static_cast<uint32_t>(std::min(7.0f, std::round(std::ldexp(magnitude, 9))));
        return sign | static_cast<uint8_t>(mantissa);
    }

    const int exponent = std::ilogb(magnitude);
    uint32_t encoded_exponent = static_cast<uint32_t>(exponent + 7);
    uint32_t mantissa = static_cast<uint32_t>(std::round((std::ldexp(magnitude, -exponent) - 1.0f) * 8.0f));
    if (mantissa == 8)
    {
        mantissa = 0;
        ++encoded_exponent;
    }
    if (encoded_exponent >= 15)
    {
        encoded_exponent = 15;
        mantissa = std::min(mantissa, 6u);
    }
    return sign | static_cast<uint8_t>(encoded_exponent << 3 | mantissa);
}

static float scalar_float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept
{
    const std::array<float, 256>& table = float8_table();
    float result = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        float partial = 0.0f;
        for (uint32_t index = block_begin; index < block_end; ++index)
            partial += table[weights[index]] * input[index];
        result += partial * scales[block];
    }
    return result;
}

static void scalar_float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count,
                                               uint32_t block_size, uint32_t row_count, float* output) noexcept
{
    for (uint32_t row = 0; row < row_count; ++row)
    {
        output[row] = scalar_float8_e4m3_block_dot(weights + static_cast<size_t>(row) * weight_row_stride, scales, input, count, block_size);
    }
}

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma"))) static float horizontal_sum(__m256 values) noexcept
{
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(values), _mm256_extractf128_ps(values, 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

__attribute__((target("avx2,fma"))) static __m256 decode_float8_e4m3_avx2(__m128i packed) noexcept
{
    const __m256i bytes = _mm256_cvtepu8_epi32(packed);
    const __m256i exponent = _mm256_and_si256(_mm256_srli_epi32(bytes, 3), _mm256_set1_epi32(15));
    const __m256i mantissa = _mm256_and_si256(bytes, _mm256_set1_epi32(7));
    const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(bytes, _mm256_set1_epi32(128)), 24);

    const __m256i normal_bits = _mm256_or_si256(
        sign,
        _mm256_or_si256(
            _mm256_slli_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(120)), 23),
            _mm256_slli_epi32(mantissa, 20)));
    const __m256 normal = _mm256_castsi256_ps(normal_bits);
    const __m256 subnormal_magnitude = _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa), _mm256_set1_ps(1.0f / 512.0f));
    const __m256 subnormal = _mm256_xor_ps(subnormal_magnitude, _mm256_castsi256_ps(sign));
    const __m256 exponent_zero = _mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
    __m256 decoded = _mm256_blendv_ps(normal, subnormal, exponent_zero);

    const __m256i nan_mask = _mm256_and_si256(
        _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(15)),
        _mm256_cmpeq_epi32(mantissa, _mm256_set1_epi32(7)));
    const __m256 nan_value = _mm256_castsi256_ps(_mm256_or_si256(sign, _mm256_set1_epi32(0x7fc00000)));
    decoded = _mm256_blendv_ps(decoded, nan_value, _mm256_castsi256_ps(nan_mask));
    return decoded;
}

__attribute__((target("avx2,fma"))) static float avx2_float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept
{
    float result = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m256 accumulator = _mm256_setzero_ps();
        uint32_t index = block_begin;
        for (; index + 8 <= block_end; index += 8)
        {
            const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights + index));
            accumulator = _mm256_fmadd_ps(decode_float8_e4m3_avx2(packed), _mm256_loadu_ps(input + index), accumulator);
        }
        float partial = horizontal_sum(accumulator);
        for (; index < block_end; ++index)
            partial += float8_e4m3_to_float(weights[index]) * input[index];
        result += partial * scales[block];
    }
    return result;
}

__attribute__((target("avx2,fma"))) static void avx2_float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count, uint32_t block_size, uint32_t row_count, float* output) noexcept
{
    if (row_count == 0 || row_count > 4)
        return;
    for (uint32_t row = 0; row < row_count; ++row)
        output[row] = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m256 accumulators[4] = {
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        };
        uint32_t index = block_begin;
        for (; index + 8 <= block_end; index += 8)
        {
            const __m256 input_values = _mm256_loadu_ps(input + index);
            for (uint32_t row = 0; row < row_count; ++row)
            {
                const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights + static_cast<size_t>(row) * weight_row_stride + index));
                accumulators[row] = _mm256_fmadd_ps(decode_float8_e4m3_avx2(packed), input_values, accumulators[row]);
            }
        }
        for (uint32_t row = 0; row < row_count; ++row)
        {
            float partial = horizontal_sum(accumulators[row]);
            for (uint32_t tail = index; tail < block_end; ++tail)
                partial += float8_e4m3_to_float(weights[static_cast<size_t>(row) * weight_row_stride + tail]) * input[tail];
            output[row] += partial * scales[block];
        }
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) static __m512 decode_float8_e4m3_avx512(__m128i packed) noexcept
{
    const __m512i bytes = _mm512_cvtepu8_epi32(packed);
    const __m512i exponent = _mm512_and_si512(_mm512_srli_epi32(bytes, 3), _mm512_set1_epi32(15));
    const __m512i mantissa = _mm512_and_si512(bytes, _mm512_set1_epi32(7));
    const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(bytes, _mm512_set1_epi32(128)), 24);

    const __m512i normal_bits = _mm512_or_si512(
        sign,
        _mm512_or_si512(
            _mm512_slli_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(120)), 23),
            _mm512_slli_epi32(mantissa, 20)));
    const __m512 normal = _mm512_castsi512_ps(normal_bits);
    const __m512 subnormal_magnitude = _mm512_mul_ps(_mm512_cvtepi32_ps(mantissa), _mm512_set1_ps(1.0f / 512.0f));
    const __m512 subnormal = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(subnormal_magnitude), sign));
    const __mmask16 exponent_zero = _mm512_cmpeq_epi32_mask(exponent, _mm512_setzero_si512());
    __m512 decoded = _mm512_mask_blend_ps(exponent_zero, normal, subnormal);

    const __mmask16 nan_mask = _mm512_cmpeq_epi32_mask(exponent, _mm512_set1_epi32(15))
                               & _mm512_cmpeq_epi32_mask(mantissa, _mm512_set1_epi32(7));
    const __m512 nan_value = _mm512_castsi512_ps(_mm512_or_si512(sign, _mm512_set1_epi32(0x7fc00000)));
    decoded = _mm512_mask_blend_ps(nan_mask, decoded, nan_value);
    return decoded;
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) static float avx512_float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept
{
    float result = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 accumulator = _mm512_setzero_ps();
        uint32_t index = block_begin;
        for (; index + 16 <= block_end; index += 16)
        {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
            accumulator = _mm512_fmadd_ps(decode_float8_e4m3_avx512(packed), _mm512_loadu_ps(input + index), accumulator);
        }
        float partial = _mm512_reduce_add_ps(accumulator);
        for (; index < block_end; ++index)
            partial += float8_e4m3_to_float(weights[index]) * input[index];
        result += partial * scales[block];
    }
    return result;
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma"))) static void avx512_float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count, uint32_t block_size, uint32_t row_count, float* output) noexcept
{
    if (row_count == 0 || row_count > 4)
        return;
    for (uint32_t row = 0; row < row_count; ++row)
        output[row] = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 accumulators[4] = {
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
        };
        uint32_t index = block_begin;
        for (; index + 16 <= block_end; index += 16)
        {
            const __m512 input_values = _mm512_loadu_ps(input + index);
            for (uint32_t row = 0; row < row_count; ++row)
            {
                const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + static_cast<size_t>(row) * weight_row_stride + index));
                accumulators[row] = _mm512_fmadd_ps(decode_float8_e4m3_avx512(packed), input_values, accumulators[row]);
            }
        }
        for (uint32_t row = 0; row < row_count; ++row)
        {
            float partial = _mm512_reduce_add_ps(accumulators[row]);
            for (uint32_t tail = index; tail < block_end; ++tail)
                partial += float8_e4m3_to_float(weights[static_cast<size_t>(row) * weight_row_stride + tail]) * input[tail];
            output[row] += partial * scales[block];
        }
    }
}
#endif

enum class Float8KernelKind
{
    Scalar,
    X86Avx2,
    X86Avx512
};

using Float8DotFunction = float (*)(const uint8_t*, const float*, const float*, uint32_t, uint32_t) noexcept;
using Float8Rows4Function = void (*)(const uint8_t*, uint32_t, const float*, const float*, uint32_t, uint32_t, uint32_t, float*) noexcept;

struct Float8KernelDispatch
{
    Float8KernelKind kind = Float8KernelKind::Scalar;
    Float8DotFunction function = scalar_float8_e4m3_block_dot;
    Float8Rows4Function rows4 = scalar_float8_e4m3_block_dot_rows4;
};

static Float8KernelDispatch select_float8_kernel() noexcept
{
    std::array<Float8KernelDispatch, 3> candidates = {};
    size_t candidate_count = 0;
    candidates[candidate_count++] = {};
    const uint64_t isa = detect_cpu_isa_capabilities().flags;
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        candidates[candidate_count++] = {Float8KernelKind::X86Avx2, msvc_avx2_float8_e4m3_block_dot, msvc_avx2_float8_e4m3_block_dot_rows4};
    if ((isa & CpuIsaX86Avx512) != 0)
        candidates[candidate_count++] = {Float8KernelKind::X86Avx512, msvc_avx512_float8_e4m3_block_dot, msvc_avx512_float8_e4m3_block_dot_rows4};
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        candidates[candidate_count++] = {Float8KernelKind::X86Avx2, avx2_float8_e4m3_block_dot, avx2_float8_e4m3_block_dot_rows4};
    if ((isa & CpuIsaX86Avx512) != 0)
        candidates[candidate_count++] = {Float8KernelKind::X86Avx512, avx512_float8_e4m3_block_dot, avx512_float8_e4m3_block_dot_rows4};
#else
    (void)isa;
#endif

    const char* override_name = nullptr;
#if defined(_MSC_VER)
    std::array<char, 16> override_storage = {};
    size_t override_length = 0;
    if (getenv_s(&override_length, override_storage.data(), override_storage.size(), "NCNN_MOE_FLOAT8_KERNEL") == 0 && override_length > 1
        && override_length <= override_storage.size())
    {
        override_name = override_storage.data();
    }
#else
    override_name = std::getenv("NCNN_MOE_FLOAT8_KERNEL");
#endif
    if (override_name && override_name[0] != '\0')
    {
        for (size_t index = 0; index < candidate_count; ++index)
        {
            const bool selected = (std::strcmp(override_name, "scalar") == 0 && candidates[index].kind == Float8KernelKind::Scalar)
                                  || (std::strcmp(override_name, "avx2") == 0 && candidates[index].kind == Float8KernelKind::X86Avx2)
                                  || (std::strcmp(override_name, "avx512") == 0 && candidates[index].kind == Float8KernelKind::X86Avx512);
            if (selected)
                return candidates[index];
        }
    }
    return candidates[candidate_count - 1];
}

static const Float8KernelDispatch& float8_kernel_dispatch() noexcept
{
    static const Float8KernelDispatch dispatch = select_float8_kernel();
    return dispatch;
}

float float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept
{
    return float8_kernel_dispatch().function(weights, scales, input, count, block_size);
}

void float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count, uint32_t block_size,
                                 uint32_t row_count, float* output) noexcept
{
    float8_kernel_dispatch().rows4(weights, weight_row_stride, scales, input, count, block_size, row_count, output);
}

const char* float8_kernel_name() noexcept
{
    switch (float8_kernel_dispatch().kind)
    {
    case Float8KernelKind::X86Avx2: return "x86-avx2-fma";
    case Float8KernelKind::X86Avx512: return "x86-avx512";
    case Float8KernelKind::Scalar: return "scalar";
    }
    return "scalar";
}

static uint32_t select_float8_linear_row_group_size() noexcept
{
    const char* override_name = nullptr;
#if defined(_MSC_VER)
    std::array<char, 8> override_storage = {};
    size_t override_length = 0;
    if (getenv_s(&override_length, override_storage.data(), override_storage.size(), "NCNN_MOE_FLOAT8_ROW_GROUP") == 0 && override_length > 1
        && override_length <= override_storage.size())
    {
        override_name = override_storage.data();
    }
#else
    override_name = std::getenv("NCNN_MOE_FLOAT8_ROW_GROUP");
#endif
    if (override_name)
    {
        if (std::strcmp(override_name, "1") == 0)
            return 1;
        if (std::strcmp(override_name, "2") == 0)
            return 2;
        if (std::strcmp(override_name, "4") == 0)
            return 4;
    }
    return 4;
}

uint32_t float8_linear_row_group_size() noexcept
{
    static const uint32_t group_size = select_float8_linear_row_group_size();
    return group_size;
}

void quantize_float8_e4m3_inplace(float* values, uint32_t count, uint32_t block_size, bool power_of_two_scale) noexcept
{
    for (uint32_t block_begin = 0; block_begin < count; block_begin += block_size)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        float maximum = 1e-4f;
        for (uint32_t index = block_begin; index < block_end; ++index)
            maximum = std::max(maximum, std::fabs(values[index]));
        float scale = maximum / 448.0f;
        if (power_of_two_scale)
            scale = std::exp2(std::ceil(std::log2(scale)));
        for (uint32_t index = block_begin; index < block_end; ++index)
        {
            const float normalized = std::clamp(values[index] / scale, -448.0f, 448.0f);
            values[index] = float8_e4m3_to_float(float_to_float8_e4m3(normalized)) * scale;
        }
    }
}

void quantize_float4_e2m1_inplace(float* values, uint32_t count, uint32_t block_size) noexcept
{
    static constexpr std::array<float, 8> magnitudes = {
        0.0f,
        0.5f,
        1.0f,
        1.5f,
        2.0f,
        3.0f,
        4.0f,
        6.0f,
    };
    for (uint32_t block_begin = 0; block_begin < count; block_begin += block_size)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        float maximum = 6.0f * std::numeric_limits<float>::min();
        for (uint32_t index = block_begin; index < block_end; ++index)
            maximum = std::max(maximum, std::fabs(values[index]));
        const float scale = std::exp2(std::ceil(std::log2(maximum / 6.0f)));
        for (uint32_t index = block_begin; index < block_end; ++index)
        {
            const float normalized = std::clamp(std::fabs(values[index] / scale), 0.0f, 6.0f);
            float selected = magnitudes.front();
            float distance = std::fabs(normalized - selected);
            for (float candidate : magnitudes)
            {
                const float candidate_distance = std::fabs(normalized - candidate);
                if (candidate_distance < distance)
                {
                    selected = candidate;
                    distance = candidate_distance;
                }
            }
            values[index] = std::copysign(selected * scale, values[index]);
        }
    }
}

} // namespace moe
} // namespace ncnn
