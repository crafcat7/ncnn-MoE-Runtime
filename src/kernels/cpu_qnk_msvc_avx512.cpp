#include "cpu_qnk_msvc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace ncnn {
namespace moe {

#if defined(NCNN_MOE_MSVC_X86_SIMD)
static float horizontal_sum(__m512 value) noexcept
{
    __m256 halves = _mm256_add_ps(_mm512_castps512_ps256(value), _mm512_extractf32x8_ps(value, 1));
    __m128 low = _mm256_castps256_ps128(halves);
    __m128 high = _mm256_extractf128_ps(halves, 1);
    const __m128 pairs = _mm_add_ps(low, high);
    const __m128 shuffled = _mm_add_ps(pairs, _mm_movehl_ps(pairs, pairs));
    return _mm_cvtss_f32(_mm_add_ss(shuffled, _mm_movehdup_ps(shuffled)));
}

static float dot_decoded(const float* decoded, const float* input) noexcept
{
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    for (uint32_t offset = 0; offset < qnk_block_elements; offset += 64)
    {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(decoded + offset), _mm512_loadu_ps(input + offset), sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(decoded + offset + 16), _mm512_loadu_ps(input + offset + 16), sum1);
        sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(decoded + offset + 32), _mm512_loadu_ps(input + offset + 32), sum2);
        sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(decoded + offset + 48), _mm512_loadu_ps(input + offset + 48), sum3);
    }
    return horizontal_sum(_mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3)));
}

static float half_to_float(uint16_t value) noexcept
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = sign;
    if (exponent == 0)
    {
        if (mantissa != 0)
        {
            uint32_t normalized = mantissa;
            int32_t exponent_shift = -1;
            while ((normalized & 0x0400u) == 0)
            {
                normalized <<= 1;
                --exponent_shift;
            }
            normalized &= 0x03ffu;
            bits |= static_cast<uint32_t>(127 - 15 + 1 + exponent_shift) << 23;
            bits |= normalized << 13;
        }
    }
    else if (exponent == 0x1fu)
    {
        bits |= 0x7f800000u | (mantissa << 13);
    }
    else
    {
        bits |= (exponent + (127 - 15)) << 23;
        bits |= mantissa << 13;
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static float load_f16(const uint8_t* source) noexcept
{
    uint16_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return half_to_float(value);
}

static void get_scale_min_k4(
    int index,
    const uint8_t* scales,
    uint8_t& scale,
    uint8_t& minimum) noexcept
{
    if (index < 4)
    {
        scale = scales[index] & 63u;
        minimum = scales[index + 4] & 63u;
    }
    else
    {
        scale = static_cast<uint8_t>((scales[index + 4] & 0x0fu) | ((scales[index - 4] >> 6) << 4));
        minimum = static_cast<uint8_t>((scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
    }
}

static __m128i load_u8x16(const uint8_t* source) noexcept
{
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
}

static float dot_affine_u8x16(
    __m128i values,
    float scale,
    float minimum,
    const float* input) noexcept
{
    const __m512 value_float = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(values));
    const __m512 scaled = _mm512_fmadd_ps(
        value_float,
        _mm512_set1_ps(scale),
        _mm512_set1_ps(-minimum));
    return horizontal_sum(_mm512_mul_ps(scaled, _mm512_loadu_ps(input)));
}

static float dot_q6_u8x16(__m128i values, const float* input) noexcept
{
    const __m512i offset = _mm512_set1_epi32(32);
    const __m512i signed_values = _mm512_sub_epi32(_mm512_cvtepu8_epi32(values), offset);
    return horizontal_sum(_mm512_mul_ps(
        _mm512_cvtepi32_ps(signed_values),
        _mm512_loadu_ps(input)));
}

static __m128i q5_values(__m128i low_bits, __m128i high_bits, uint8_t mask) noexcept
{
    const __m128i low_nibbles = _mm_and_si128(low_bits, _mm_set1_epi8(0x0f));
    const __m128i selected_high_bits = _mm_and_si128(high_bits, _mm_set1_epi8(static_cast<char>(mask)));
    const __m128i is_zero = _mm_cmpeq_epi8(selected_high_bits, _mm_setzero_si128());
    const __m128i high_values = _mm_andnot_si128(is_zero, _mm_set1_epi8(16));
    return _mm_add_epi8(low_nibbles, high_values);
}

static __m128i q6_high_values(__m128i high_bits, int shift) noexcept
{
    const __m128i source_mask = _mm_set1_epi8(static_cast<char>(0x03u << shift));
    return _mm_and_si128(
        _mm_srli_epi16(_mm_and_si128(high_bits, source_mask), shift),
        _mm_set1_epi8(3));
}

static __m128i q6_values(__m128i low_bits, __m128i high_bits, int low_shift, int high_shift) noexcept
{
    const __m128i low = _mm_and_si128(
        low_shift == 0 ? low_bits : _mm_srli_epi16(low_bits, low_shift),
        _mm_set1_epi8(0x0f));
    const __m128i high = _mm_slli_epi16(q6_high_values(high_bits, high_shift), 4);
    return _mm_or_si128(low, high);
}

static float dot_q4_block_direct(const uint8_t* block, const float* input) noexcept
{
    const float d = load_f16(block);
    const float minimum = load_f16(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* values = block + 16;
    float sum = 0.0f;
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    for (uint32_t chunk = 0; chunk < 4; ++chunk)
    {
        uint8_t first_scale_value = 0;
        uint8_t first_minimum_value = 0;
        uint8_t second_scale_value = 0;
        uint8_t second_minimum_value = 0;
        get_scale_min_k4(static_cast<int>(chunk * 2), scales, first_scale_value, first_minimum_value);
        get_scale_min_k4(static_cast<int>(chunk * 2 + 1), scales, second_scale_value, second_minimum_value);
        const float first_scale = d * static_cast<float>(first_scale_value);
        const float first_minimum = minimum * static_cast<float>(first_minimum_value);
        const float second_scale = d * static_cast<float>(second_scale_value);
        const float second_minimum = minimum * static_cast<float>(second_minimum_value);
        const uint32_t input_offset = chunk * 64;
        for (uint32_t lane = 0; lane < 32; lane += 16)
        {
            const __m128i encoded = load_u8x16(values + chunk * 32 + lane);
            const __m128i low = _mm_and_si128(encoded, nibble_mask);
            const __m128i high = _mm_and_si128(_mm_srli_epi16(encoded, 4), nibble_mask);
            const float* first_input = input + input_offset + lane;
            const float* second_input = first_input + 32;
            sum += dot_affine_u8x16(low, first_scale, first_minimum, first_input);
            sum += dot_affine_u8x16(high, second_scale, second_minimum, second_input);
        }
    }
    return sum;
}

static float dot_q5_block_direct(const uint8_t* block, const float* input) noexcept
{
    const float d = load_f16(block);
    const float minimum = load_f16(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* high_bits = block + 16;
    const uint8_t* values = block + 48;
    float sum = 0.0f;
    uint8_t first_mask = 1;
    uint8_t second_mask = 2;
    for (uint32_t chunk = 0; chunk < 4; ++chunk)
    {
        uint8_t first_scale_value = 0;
        uint8_t first_minimum_value = 0;
        uint8_t second_scale_value = 0;
        uint8_t second_minimum_value = 0;
        get_scale_min_k4(static_cast<int>(chunk * 2), scales, first_scale_value, first_minimum_value);
        get_scale_min_k4(static_cast<int>(chunk * 2 + 1), scales, second_scale_value, second_minimum_value);
        const float first_scale = d * static_cast<float>(first_scale_value);
        const float first_minimum = minimum * static_cast<float>(first_minimum_value);
        const float second_scale = d * static_cast<float>(second_scale_value);
        const float second_minimum = minimum * static_cast<float>(second_minimum_value);
        const uint32_t input_offset = chunk * 64;
        for (uint32_t lane = 0; lane < 32; lane += 16)
        {
            const __m128i encoded = load_u8x16(values + chunk * 32 + lane);
            const __m128i high = load_u8x16(high_bits + lane);
            const float* first_input = input + input_offset + lane;
            const float* second_input = first_input + 32;
            sum += dot_affine_u8x16(q5_values(encoded, high, first_mask), first_scale, first_minimum, first_input);
            sum += dot_affine_u8x16(q5_values(_mm_srli_epi16(encoded, 4), high, second_mask), second_scale, second_minimum, second_input);
        }
        first_mask = static_cast<uint8_t>(first_mask << 2);
        second_mask = static_cast<uint8_t>(second_mask << 2);
    }
    return sum;
}

static float dot_q6_block_direct(const uint8_t* block, const float* input) noexcept
{
    const float d = load_f16(block + 208);
    float sum = 0.0f;
    for (uint32_t half = 0; half < 2; ++half)
    {
        const uint8_t* low_bits = block + half * 64;
        const uint8_t* high_bits = block + 128 + half * 32;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192 + half * 8);
        const float* input_half = input + half * 128;
        for (uint32_t lane = 0; lane < 32; lane += 16)
        {
            const __m128i high = load_u8x16(high_bits + lane);
            const __m128i first_low = load_u8x16(low_bits + lane);
            const __m128i second_low = load_u8x16(low_bits + 32 + lane);
            const float* first_input = input_half + lane;
            const float* second_input = first_input + 32;
            const float* third_input = first_input + 64;
            const float* fourth_input = first_input + 96;
            sum += d * static_cast<float>(scales[0]) * dot_q6_u8x16(q6_values(first_low, high, 0, 0), first_input);
            sum += d * static_cast<float>(scales[2]) * dot_q6_u8x16(q6_values(second_low, high, 0, 2), second_input);
            sum += d * static_cast<float>(scales[4]) * dot_q6_u8x16(q6_values(first_low, high, 4, 4), third_input);
            sum += d * static_cast<float>(scales[6]) * dot_q6_u8x16(q6_values(second_low, high, 4, 6), fourth_input);
        }
    }
    return sum;
}

float msvc_avx512_qnk_dot_block(DType dtype, const uint8_t* block, const float* input) noexcept
{
    if (!block || !input)
        return 0.0f;
    if (dtype == DType::Q4K)
        return dot_q4_block_direct(block, input);
    if (dtype == DType::Q5K)
        return dot_q5_block_direct(block, input);
    if (dtype == DType::Q6K)
        return dot_q6_block_direct(block, input);
    return msvc_avx2_qnk_dot_block(dtype, block, input);
}

static float horizontal_max(__m512 value) noexcept
{
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, value);
    float maximum = lanes[0];
    for (uint32_t index = 1; index < 16; ++index)
        maximum = std::max(maximum, lanes[index]);
    return maximum;
}

void msvc_avx512_qnk_gemm(
    const QnKPack& weights,
    const float* input,
    size_t input_stride,
    size_t token_count,
    float* output,
    size_t output_stride) noexcept
{
    alignas(64) float decoded[qnk_block_elements];
    for (size_t row = 0; row < weights.rows; ++row)
    {
        for (size_t token = 0; token < token_count; ++token)
        {
            const float* token_input = input + token * input_stride;
            float sum = 0.0f;
            for (uint32_t block = 0; block < weights.block_count; ++block)
            {
                const uint8_t* encoded = qnk_packed_block(weights, row, block);
                const float* block_input = token_input + static_cast<size_t>(block) * qnk_block_elements;
                if (is_qnk_dtype(weights.dtype))
                    sum += msvc_avx512_qnk_dot_block(weights.dtype, encoded, block_input);
                else
                {
                    qnk_dequantize_block(weights.dtype, encoded, decoded);
                    sum += dot_decoded(decoded, block_input);
                }
            }
            output[token * output_stride + row] = sum;
        }
    }
}

void msvc_avx512_qnk_q8k_quantize(const float* source, uint8_t* output, uint32_t columns) noexcept
{
    const uint32_t block_count = columns / qnk_block_elements;
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);
    const __m512 lower = _mm512_set1_ps(-127.0f);
    const __m512 upper = _mm512_set1_ps(127.0f);
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const float* values = source + static_cast<size_t>(block_index) * qnk_block_elements;
        uint8_t* encoded = output + static_cast<size_t>(block_index) * qnk_block_bytes(DType::Q8K);
        __m512 maximum_values = _mm512_setzero_ps();
        for (uint32_t index = 0; index < qnk_block_elements; index += 16)
        {
            const __m512 current = _mm512_loadu_ps(values + index);
            maximum_values = _mm512_max_ps(maximum_values, _mm512_andnot_ps(sign_mask, current));
        }
        const float maximum = horizontal_max(maximum_values);
        if (maximum == 0.0f)
        {
            std::memset(encoded, 0, qnk_block_bytes(DType::Q8K));
            continue;
        }
        float signed_maximum = 0.0f;
        for (uint32_t index = 0; index < qnk_block_elements; ++index)
        {
            if (std::fabs(values[index]) == maximum)
            {
                signed_maximum = values[index];
                break;
            }
        }
        const float inverse_scale_value = -127.0f / signed_maximum;
        const float scale = 1.0f / inverse_scale_value;
        std::memcpy(encoded, &scale, sizeof(scale));
        int8_t* quantized = reinterpret_cast<int8_t*>(encoded + 4);
        const __m512 inverse_scale = _mm512_set1_ps(inverse_scale_value);
        for (uint32_t index = 0; index < qnk_block_elements; index += 16)
        {
            __m512 normalized = _mm512_mul_ps(_mm512_loadu_ps(values + index), inverse_scale);
            normalized = _mm512_max_ps(lower, _mm512_min_ps(upper, normalized));
            const __m512i converted = _mm512_cvtps_epi32(normalized);
            alignas(64) int32_t temporary[16];
            _mm512_store_si512(reinterpret_cast<__m512i*>(temporary), converted);
            for (uint32_t lane = 0; lane < 16; ++lane)
                quantized[index + lane] = static_cast<int8_t>(temporary[lane]);
        }
        for (uint32_t sum_index = 0; sum_index < qnk_block_elements / 16; ++sum_index)
        {
            int16_t sum = 0;
            for (uint32_t lane = 0; lane < 16; ++lane)
                sum = static_cast<int16_t>(sum + quantized[sum_index * 16 + lane]);
            std::memcpy(encoded + 260 + sum_index * sizeof(int16_t), &sum, sizeof(sum));
        }
    }
}
#endif

} // namespace moe
} // namespace ncnn
