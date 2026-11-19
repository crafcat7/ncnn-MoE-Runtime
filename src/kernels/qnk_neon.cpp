// ARM64 int8 activation path for the 32-value GGML QnK formats.
//
// The x86 build has an opt-in Q8-activation path for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/
// IQ4_NL: quantize one activation row to int8 blocks, then keep the weight
// codes packed and use integer dot products. The arm64 build previously fell
// back to dequantize-to-fp32 + FMA for every block. This file provides the
// same contract with NEON, using SDOT when the CPU advertises dotprod.

#include "qnk.h"
#include "ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace ncnn {
namespace moe {

namespace {

constexpr int8_t kIq4NlTable[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline float load_f16(const uint8_t* data) noexcept
{
    uint16_t bits = 0;
    std::memcpy(&bits, data, sizeof(bits));
    __fp16 half = 0.0f;
    std::memcpy(&half, &bits, sizeof(half));
    return static_cast<float>(half);
}

inline int8x16_t low_nibbles(uint8x16_t packed) noexcept
{
    return vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0f)));
}

inline int8x16_t high_nibbles(uint8x16_t packed) noexcept
{
    return vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
}

// Expand the four high-bit bytes of a Q5 block into 0x10 in the matching lane.
inline uint8x16_t q5_high_bits(const uint8_t* high, uint32_t half) noexcept
{
    alignas(16) static constexpr uint8_t lane_mask[16] = {
        1, 2, 4, 8, 16, 32, 64, static_cast<uint8_t>(0x80),
        1, 2, 4, 8, 16, 32, 64, static_cast<uint8_t>(0x80)};
    const uint8x16_t masks = vld1q_u8(lane_mask);
    const uint8x16_t bytes = vcombine_u8(vdup_n_u8(high[half * 2]),
                                          vdup_n_u8(high[half * 2 + 1]));
    return vandq_u8(vceqq_u8(vandq_u8(bytes, masks), masks), vdupq_n_u8(0x10));
}

#if defined(__clang__)
#define NCNN_MOE_ARM_DOTPROD __attribute__((target("+dotprod")))
#else
#define NCNN_MOE_ARM_DOTPROD
#endif

#if defined(__clang__)
NCNN_MOE_ARM_DOTPROD __attribute__((always_inline)) inline int32x4_t dot32_vec(int8x16_t first,
                                              int8x16_t second,
                                              const int8_t* input) noexcept
{
    int32x4_t accumulator = vdupq_n_s32(0);
    accumulator = vdotq_s32(accumulator, first, vld1q_s8(input));
    accumulator = vdotq_s32(accumulator, second, vld1q_s8(input + 16));
    return accumulator;
}

NCNN_MOE_ARM_DOTPROD __attribute__((always_inline)) inline int32_t dot32(int8x16_t first,
                                          int8x16_t second,
                                          const int8_t* input) noexcept
{
    return vaddvq_s32(dot32_vec(first, second, input));
}
#endif

} // namespace

bool neon_qnk_q8_supported() noexcept
{
#if defined(__clang__) && defined(__APPLE__)
    int value = 0;
    std::size_t size = sizeof(value);
    return sysctlbyname("hw.optional.arm.FEAT_DotProd", &value, &size, nullptr, 0) == 0
           && value != 0;
#elif defined(__clang__) && defined(__linux__) && defined(HWCAP_ASIMDDP)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}


void neon_qnk_q8_quant_pack4_group(const float* source,
                                   std::size_t group,
                                   uint32_t columns,
                                   int8_t* pack,
                                   float* scales) noexcept
{
    const uint32_t block_count = columns / 32;
    for (uint32_t block = 0; block < block_count; ++block)
    {
        int8_t* destination = pack + (group * block_count + block) * 128;
        for (uint32_t token = 0; token < 4; ++token)
        {
            const float* values = source
                                  + (group * 4 + token) * static_cast<std::size_t>(columns)
                                  + static_cast<std::size_t>(block) * 32;
            float32x4_t maxima = vabsq_f32(vld1q_f32(values));
            maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(values + 4)));
            maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(values + 8)));
            maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(values + 12)));
            maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(values + 16)));
            maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(values + 20)));
            maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(values + 24)));
            maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(values + 28)));
            const float maximum = vmaxvq_f32(maxima);
            const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
            scales[(group * 4 + token) * block_count + block] = scale;
            const float32x4_t inverse = vdupq_n_f32(1.0f / scale);
            for (uint32_t chunk = 0; chunk < 4; ++chunk)
            {
                const uint32_t index = chunk * 8;
                float32x4_t first = vmulq_f32(vld1q_f32(values + index), inverse);
                float32x4_t second = vmulq_f32(vld1q_f32(values + index + 4), inverse);
                first = vmaxq_f32(vdupq_n_f32(-127.0f), vminq_f32(vdupq_n_f32(127.0f), first));
                second = vmaxq_f32(vdupq_n_f32(-127.0f), vminq_f32(vdupq_n_f32(127.0f), second));
                const int32x4_t first_int = vcvtnq_s32_f32(first);
                const int32x4_t second_int = vcvtnq_s32_f32(second);
                const int16x8_t narrow = vcombine_s16(vqmovn_s32(first_int), vqmovn_s32(second_int));
                vst1_s8(destination + chunk * 32 + token * 8, vqmovn_s16(narrow));
            }
        }
    }
}

void neon_qnk_q8_quant_pack4(const float* source,
                              std::size_t tokens,
                              uint32_t columns,
                              int8_t* pack,
                              float* scales) noexcept
{
    if (!source || !pack || !scales || tokens == 0 || tokens % 4 != 0
        || columns == 0 || columns % 32 != 0)
        return;
    const std::size_t groups = tokens / 4;
    const int team_size = omp_in_parallel() != 0
                              ? 1
                              : static_cast<int>(std::min<std::size_t>(groups, static_cast<std::size_t>(cpu_linear_num_threads())));
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
    for (int64_t group = 0; group < static_cast<int64_t>(groups); ++group)
        neon_qnk_q8_quant_pack4_group(source, static_cast<std::size_t>(group), columns, pack, scales);
}

void neon_qnk_q8_quant(const float* source,
                       int8_t* values,
                       float* scales,
                       int32_t* sums,
                       uint32_t columns) noexcept
{
    if (!source || !values || !scales || !sums || columns == 0 || columns % 32 != 0)
        return;
    const uint32_t block_count = columns / 32;
    const float32x4_t lower = vdupq_n_f32(-127.0f);
    const float32x4_t upper = vdupq_n_f32(127.0f);
    for (uint32_t block = 0; block < block_count; ++block)
    {
        const float* block_source = source + static_cast<std::size_t>(block) * 32;
        float32x4_t maxima = vabsq_f32(vld1q_f32(block_source));
        maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(block_source + 4)));
        maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(block_source + 8)));
        maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(block_source + 12)));
        maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(block_source + 16)));
        maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(block_source + 20)));
        maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(block_source + 24)));
        maxima = vmaxq_f32(maxima, vabsq_f32(vld1q_f32(block_source + 28)));
        const float maximum = vmaxvq_f32(maxima);
        const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        scales[block] = scale;
        const float32x4_t inverse = vdupq_n_f32(1.0f / scale);
        int8_t* block_values = values + static_cast<std::size_t>(block) * 32;
        int32x4_t sum_vector = vdupq_n_s32(0);
        for (uint32_t index = 0; index < 32; index += 8)
        {
            float32x4_t first = vmulq_f32(vld1q_f32(block_source + index), inverse);
            float32x4_t second = vmulq_f32(vld1q_f32(block_source + index + 4), inverse);
            first = vmaxq_f32(lower, vminq_f32(upper, first));
            second = vmaxq_f32(lower, vminq_f32(upper, second));
            const int32x4_t first_int = vcvtnq_s32_f32(first);
            const int32x4_t second_int = vcvtnq_s32_f32(second);
            sum_vector = vaddq_s32(sum_vector, vaddq_s32(first_int, second_int));
            const int16x8_t narrow = vcombine_s16(vqmovn_s32(first_int), vqmovn_s32(second_int));
            vst1_s8(block_values + index, vqmovn_s16(narrow));
        }
        sums[block] = vaddvq_s32(sum_vector);
    }
}

#if defined(__clang__)
NCNN_MOE_ARM_DOTPROD static float q8_0_row(const uint8_t* weights,
                                            std::size_t block_stride,
                                            uint32_t block_count,
                                            const int8_t* input,
                                            const float* input_scales) noexcept
{
    float32x4_t accumulators[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                   vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    uint32_t block = 0;
    for (; block + 3 < block_count; block += 4)
    {
#pragma clang loop unroll(full)
        for (uint32_t lane = 0; lane < 4; ++lane)
        {
            const uint8_t* encoded = weights + static_cast<std::size_t>(block + lane) * block_stride;
            const int8_t* block_input = input + static_cast<std::size_t>(block + lane) * 32;
            const int32x4_t dot = dot32_vec(vld1q_s8(reinterpret_cast<const int8_t*>(encoded + 2)),
                                            vld1q_s8(reinterpret_cast<const int8_t*>(encoded + 18)),
                                            block_input);
            accumulators[lane] = vmlaq_n_f32(accumulators[lane], vcvtq_f32_s32(dot),
                                             load_f16(encoded) * input_scales[block + lane]);
        }
    }
    for (; block + 1 < block_count; block += 2)
    {
        const uint8_t* encoded0 = weights + static_cast<std::size_t>(block) * block_stride;
        const uint8_t* encoded1 = encoded0 + block_stride;
        const int8_t* input0 = input + static_cast<std::size_t>(block) * 32;
        const int8_t* input1 = input0 + 32;
        const int32x4_t dot0 = dot32_vec(vld1q_s8(reinterpret_cast<const int8_t*>(encoded0 + 2)),
                                         vld1q_s8(reinterpret_cast<const int8_t*>(encoded0 + 18)),
                                         input0);
        const int32x4_t dot1 = dot32_vec(vld1q_s8(reinterpret_cast<const int8_t*>(encoded1 + 2)),
                                         vld1q_s8(reinterpret_cast<const int8_t*>(encoded1 + 18)),
                                         input1);
        accumulators[0] = vmlaq_n_f32(accumulators[0], vcvtq_f32_s32(dot0),
                                      load_f16(encoded0) * input_scales[block]);
        accumulators[1] = vmlaq_n_f32(accumulators[1], vcvtq_f32_s32(dot1),
                                      load_f16(encoded1) * input_scales[block + 1]);
    }
    for (; block < block_count; ++block)
    {
        const uint8_t* encoded = weights + static_cast<std::size_t>(block) * block_stride;
        const int8_t* block_input = input + static_cast<std::size_t>(block) * 32;
        const int32x4_t dot = dot32_vec(vld1q_s8(reinterpret_cast<const int8_t*>(encoded + 2)),
                                        vld1q_s8(reinterpret_cast<const int8_t*>(encoded + 18)),
                                        block_input);
        accumulators[0] = vmlaq_n_f32(accumulators[0], vcvtq_f32_s32(dot),
                                      load_f16(encoded) * input_scales[block]);
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(accumulators[0], accumulators[1]),
                                vaddq_f32(accumulators[2], accumulators[3])));
}

NCNN_MOE_ARM_DOTPROD float neon_qnk_q8_row(DType dtype,
                                            const uint8_t* weights,
                                            std::size_t block_stride,
                                            uint32_t block_count,
                                            const int8_t* input,
                                            const float* input_scales,
                                            const int32_t* input_sums) noexcept
{
    if (!weights || !input || !input_scales || !input_sums || block_count == 0)
        return 0.0f;
    if (dtype == DType::Q8_0)
        return q8_0_row(weights, block_stride, block_count, input, input_scales);
    const int8x16_t table = vld1q_s8(kIq4NlTable);
    float result = 0.0f;
    for (uint32_t block = 0; block < block_count; ++block)
    {
        const uint8_t* encoded = weights + static_cast<std::size_t>(block) * block_stride;
        const int8_t* block_input = input + static_cast<std::size_t>(block) * 32;
        int8x16_t first = vdupq_n_s8(0);
        int8x16_t second = vdupq_n_s8(0);
        float weight_scale = 0.0f;
        float additive = 0.0f;
        float code_bias = 0.0f;
        switch (dtype)
        {
        case DType::Q4_0:
        case DType::Q4_1:
        {
            const bool affine = dtype == DType::Q4_1;
            const uint8x16_t packed = vld1q_u8(encoded + (affine ? 4 : 2));
            first = low_nibbles(packed);
            second = high_nibbles(packed);
            weight_scale = load_f16(encoded);
            additive = affine ? load_f16(encoded + 2) : 0.0f;
            code_bias = affine ? 0.0f : 8.0f;
            break;
        }
        case DType::Q5_0:
        case DType::Q5_1:
        {
            const bool affine = dtype == DType::Q5_1;
            const uint8_t* high = encoded + (affine ? 4 : 2);
            const uint8x16_t packed = vld1q_u8(encoded + (affine ? 8 : 6));
            first = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(packed, vdupq_n_u8(0x0f)),
                                                 q5_high_bits(high, 0)));
            second = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(packed, 4),
                                                  q5_high_bits(high, 1)));
            weight_scale = load_f16(encoded);
            additive = affine ? load_f16(encoded + 2) : 0.0f;
            code_bias = affine ? 0.0f : 16.0f;
            break;
        }
        case DType::Q8_0:
            first = vld1q_s8(reinterpret_cast<const int8_t*>(encoded + 2));
            second = vld1q_s8(reinterpret_cast<const int8_t*>(encoded + 18));
            weight_scale = load_f16(encoded);
            break;
        case DType::IQ4NL:
        {
            const uint8x16_t packed = vld1q_u8(encoded + 2);
            first = vqtbl1q_s8(table, vandq_u8(packed, vdupq_n_u8(0x0f)));
            second = vqtbl1q_s8(table, vshrq_n_u8(packed, 4));
            weight_scale = load_f16(encoded);
            break;
        }
        default:
            return 0.0f;
        }
        const int32_t integer_dot = dot32(first, second, block_input);
        const float input_scale = input_scales[block];
        result += weight_scale * input_scale * static_cast<float>(integer_dot)
                  + (additive - weight_scale * code_bias) * input_scale
                        * static_cast<float>(input_sums[block]);
    }
    return result;
}
NCNN_MOE_ARM_DOTPROD void neon_qnk_prefill_q8_0_rows8(const uint8_t* __restrict weights,
                                                        std::size_t rows,
                                                        uint32_t columns,
                                                        const int8_t* __restrict activation_codes,
                                                        const float* __restrict activation_scales,
                                                        std::size_t token_count,
                                                        float* __restrict output,
                                                        std::size_t output_stride) noexcept
{
    if (!weights || !activation_codes || !activation_scales || !output
        || rows == 0 || token_count == 0 || columns == 0 || columns % 32 != 0
        || rows % 8 != 0)
        return;
    const uint32_t block_count = columns / 32;
    const std::size_t row_tiles = rows / 8;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        row_tiles, static_cast<std::size_t>(cpu_linear_team_size(operation_count, DType::Q8_0))));
#pragma omp parallel for schedule(dynamic, 1) num_threads(team_size) if (team_size > 1)
    for (int64_t tile_index = 0; tile_index < static_cast<int64_t>(row_tiles); ++tile_index)
    {
        const std::size_t first_row = static_cast<std::size_t>(tile_index) * 8;
        for (uint32_t block = 0; block < block_count; ++block)
        {
            int8x16_t weight_low[8];
            int8x16_t weight_high[8];
            float weight_scale[8];
            for (uint32_t row = 0; row < 8; ++row)
            {
                const uint8_t* encoded = weights
                                         + ((first_row + row) * static_cast<std::size_t>(block_count) + block) * 34;
                weight_low[row] = vld1q_s8(reinterpret_cast<const int8_t*>(encoded + 2));
                weight_high[row] = vld1q_s8(reinterpret_cast<const int8_t*>(encoded + 18));
                weight_scale[row] = load_f16(encoded);
            }
            for (std::size_t first_token = 0; first_token < token_count; first_token += 4)
            {
                const uint32_t token_count_tile = static_cast<uint32_t>(
                    std::min<std::size_t>(4, token_count - first_token));
                if (token_count_tile == 4)
                {
                    for (uint32_t pair = 0; pair < 2; ++pair)
                    {
                        const uint32_t token0 = pair * 2;
                        const uint32_t token1 = token0 + 1;
                        const int8_t* codes0 = activation_codes
                                               + (first_token + token0) * static_cast<std::size_t>(columns)
                                               + static_cast<std::size_t>(block) * 32;
                        const int8_t* codes1 = activation_codes
                                               + (first_token + token1) * static_cast<std::size_t>(columns)
                                               + static_cast<std::size_t>(block) * 32;
                        const int8x16_t activation0_low = vld1q_s8(codes0);
                        const int8x16_t activation0_high = vld1q_s8(codes0 + 16);
                        const int8x16_t activation1_low = vld1q_s8(codes1);
                        const int8x16_t activation1_high = vld1q_s8(codes1 + 16);
                        const float activation0_scale = activation_scales[(first_token + token0) * block_count + block];
                        const float activation1_scale = activation_scales[(first_token + token1) * block_count + block];
                        for (uint32_t group = 0; group < 2; ++group)
                        {
                            const float32x4_t scales = vld1q_f32(weight_scale + group * 4);
                            int32x4_t dot0[4];
                            int32x4_t dot1[4];
                            for (uint32_t row = 0; row < 4; ++row)
                            {
                                const uint32_t index = group * 4 + row;
                                dot0[row] = vdotq_s32(vdupq_n_s32(0), weight_low[index], activation0_low);
                                dot1[row] = vdotq_s32(vdupq_n_s32(0), weight_low[index], activation1_low);
                                dot0[row] = vdotq_s32(dot0[row], weight_high[index], activation0_high);
                                dot1[row] = vdotq_s32(dot1[row], weight_high[index], activation1_high);
                            }
                            const int32x4_t sums0 = vpaddq_s32(vpaddq_s32(dot0[0], dot0[1]),
                                                               vpaddq_s32(dot0[2], dot0[3]));
                            const int32x4_t sums1 = vpaddq_s32(vpaddq_s32(dot1[0], dot1[1]),
                                                               vpaddq_s32(dot1[2], dot1[3]));
                            const float32x4_t value0 = vmulq_f32(vcvtq_f32_s32(sums0),
                                                                 vmulq_n_f32(scales, activation0_scale));
                            const float32x4_t value1 = vmulq_f32(vcvtq_f32_s32(sums1),
                                                                 vmulq_n_f32(scales, activation1_scale));
                            float* destination0 = output + (first_token + token0) * output_stride + first_row + group * 4;
                            float* destination1 = output + (first_token + token1) * output_stride + first_row + group * 4;
                            if (block == 0)
                            {
                                vst1q_f32(destination0, value0);
                                vst1q_f32(destination1, value1);
                            }
                            else
                            {
                                vst1q_f32(destination0, vaddq_f32(vld1q_f32(destination0), value0));
                                vst1q_f32(destination1, vaddq_f32(vld1q_f32(destination1), value1));
                            }
                        }
                    }
                }
                else
                {
                    int8x16_t activation_low[4];
                    int8x16_t activation_high[4];
                    float activation_scale[4];
                    for (uint32_t token = 0; token < token_count_tile; ++token)
                    {
                        const int8_t* codes = activation_codes
                                              + (first_token + token) * static_cast<std::size_t>(columns)
                                              + static_cast<std::size_t>(block) * 32;
                        activation_low[token] = vld1q_s8(codes);
                        activation_high[token] = vld1q_s8(codes + 16);
                        activation_scale[token] = activation_scales[(first_token + token) * block_count + block];
                    }
                    for (uint32_t token = 0; token < token_count_tile; ++token)
                        for (uint32_t row = 0; row < 8; ++row)
                        {
                            int32x4_t dot = vdotq_s32(vdupq_n_s32(0), weight_low[row], activation_low[token]);
                            dot = vdotq_s32(dot, weight_high[row], activation_high[token]);
                            const float value = static_cast<float>(vaddvq_s32(dot))
                                                * weight_scale[row] * activation_scale[token];
                            float& destination = output[(first_token + token) * output_stride + first_row + row];
                            if (block == 0)
                                destination = value;
                            else
                                destination += value;
                        }
                }
            }
        }
    }
}

NCNN_MOE_ARM_DOTPROD void neon_qnk_prefill_q8_0(const uint8_t* weights,
                                                    std::size_t rows,
                                                    uint32_t columns,
                                                    const int8_t* activation_codes,
                                                    const float* activation_scales,
                                                    std::size_t token_count,
                                                    float* output,
                                                    std::size_t output_stride) noexcept
{
    if (!weights || !activation_codes || !activation_scales || !output
        || rows == 0 || token_count == 0 || columns == 0 || columns % 32 != 0)
        return;
    const uint32_t block_count = columns / 32;
    const std::size_t row_tiles = (rows + 3) / 4;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        row_tiles, static_cast<std::size_t>(cpu_linear_team_size(operation_count, DType::Q8_0))));
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
    for (int64_t tile_index = 0; tile_index < static_cast<int64_t>(row_tiles); ++tile_index)
    {
        const std::size_t first_row = static_cast<std::size_t>(tile_index) * 4;
        const uint32_t row_count = static_cast<uint32_t>(std::min<std::size_t>(4, rows - first_row));
        for (uint32_t block = 0; block < block_count; ++block)
        {
            int8x16_t weight_low[4];
            int8x16_t weight_high[4];
            float weight_scale[4];
            for (uint32_t row = 0; row < row_count; ++row)
            {
                const uint8_t* encoded = weights
                                         + ((first_row + row) * static_cast<std::size_t>(block_count) + block) * 34;
                weight_low[row] = vld1q_s8(reinterpret_cast<const int8_t*>(encoded + 2));
                weight_high[row] = vld1q_s8(reinterpret_cast<const int8_t*>(encoded + 18));
                weight_scale[row] = load_f16(encoded);
            }
            for (std::size_t first_token = 0; first_token < token_count; first_token += 4)
            {
                const uint32_t token_count_tile = static_cast<uint32_t>(
                    std::min<std::size_t>(4, token_count - first_token));
                int8x16_t activation_low[4];
                int8x16_t activation_high[4];
                float activation_scale[4];
                for (uint32_t token = 0; token < token_count_tile; ++token)
                {
                    const int8_t* codes = activation_codes
                                          + (first_token + token) * static_cast<std::size_t>(columns)
                                          + static_cast<std::size_t>(block) * 32;
                    activation_low[token] = vld1q_s8(codes);
                    activation_high[token] = vld1q_s8(codes + 16);
                    activation_scale[token] = activation_scales[(first_token + token) * block_count + block];
                }
                if (row_count == 4 && token_count_tile == 4)
                {
                    const float32x4_t scales = vld1q_f32(weight_scale);
                    for (uint32_t pair = 0; pair < 2; ++pair)
                    {
                        const uint32_t token0 = pair * 2;
                        const uint32_t token1 = token0 + 1;
                        int32x4_t dot0[4];
                        int32x4_t dot1[4];
                        for (uint32_t row = 0; row < 4; ++row)
                        {
                            dot0[row] = vdotq_s32(vdupq_n_s32(0), weight_low[row], activation_low[token0]);
                            dot1[row] = vdotq_s32(vdupq_n_s32(0), weight_low[row], activation_low[token1]);
                            dot0[row] = vdotq_s32(dot0[row], weight_high[row], activation_high[token0]);
                            dot1[row] = vdotq_s32(dot1[row], weight_high[row], activation_high[token1]);
                        }
                        const int32x4_t sums0 = vpaddq_s32(vpaddq_s32(dot0[0], dot0[1]),
                                                           vpaddq_s32(dot0[2], dot0[3]));
                        const int32x4_t sums1 = vpaddq_s32(vpaddq_s32(dot1[0], dot1[1]),
                                                           vpaddq_s32(dot1[2], dot1[3]));
                        const float32x4_t value0 = vmulq_f32(vcvtq_f32_s32(sums0),
                                                             vmulq_n_f32(scales, activation_scale[token0]));
                        const float32x4_t value1 = vmulq_f32(vcvtq_f32_s32(sums1),
                                                             vmulq_n_f32(scales, activation_scale[token1]));
                        float* destination0 = output + (first_token + token0) * output_stride + first_row;
                        float* destination1 = output + (first_token + token1) * output_stride + first_row;
                        if (block == 0)
                        {
                            vst1q_f32(destination0, value0);
                            vst1q_f32(destination1, value1);
                        }
                        else
                        {
                            vst1q_f32(destination0, vaddq_f32(vld1q_f32(destination0), value0));
                            vst1q_f32(destination1, vaddq_f32(vld1q_f32(destination1), value1));
                        }
                    }
                }
                else
                {
                    for (uint32_t token = 0; token < token_count_tile; ++token)
                        for (uint32_t row = 0; row < row_count; ++row)
                        {
                            int32x4_t dot = vdotq_s32(vdupq_n_s32(0), weight_low[row], activation_low[token]);
                            dot = vdotq_s32(dot, weight_high[row], activation_high[token]);
                            const float value = static_cast<float>(vaddvq_s32(dot))
                                                * weight_scale[row] * activation_scale[token];
                            float& destination = output[(first_token + token) * output_stride + first_row + row];
                            if (block == 0)
                                destination = value;
                            else
                                destination += value;
                        }
                }
            }
        }
    }
}
NCNN_MOE_ARM_DOTPROD void neon_qnk_prefill_q8_rows8(DType dtype,
                                                     const uint8_t* weights,
                                                     std::size_t rows,
                                                     uint32_t columns,
                                                     const int8_t* activation_codes,
                                                     const float* activation_scales,
                                                     const int32_t* activation_sums,
                                                     std::size_t token_count,
                                                     float* output,
                                                     std::size_t output_stride) noexcept
{
    if (!weights || !activation_codes || !activation_scales || !activation_sums || !output
        || rows == 0 || rows % 8 != 0 || token_count == 0 || columns == 0 || columns % 32 != 0)
        return;
    const int8x16_t table = vld1q_s8(kIq4NlTable);
    const uint32_t block_count = columns / 32;
    const std::size_t block_bytes = qnk_block_bytes(dtype);
    const std::size_t row_tiles = rows / 8;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        row_tiles, static_cast<std::size_t>(cpu_linear_team_size(operation_count, dtype))));
#pragma omp parallel for schedule(dynamic, 1) num_threads(team_size) if (team_size > 1)
    for (int64_t tile_index = 0; tile_index < static_cast<int64_t>(row_tiles); ++tile_index)
    {
        const std::size_t first_row = static_cast<std::size_t>(tile_index) * 8;
        for (uint32_t block = 0; block < block_count; ++block)
        {
            int8x16_t weight_low[8];
            int8x16_t weight_high[8];
            float weight_scale[8];
            float weight_additive[8];
            float weight_bias[8];
            for (uint32_t row = 0; row < 8; ++row)
            {
                const uint8_t* encoded = weights
                                         + ((first_row + row) * static_cast<std::size_t>(block_count) + block) * block_bytes;
                switch (dtype)
                {
                case DType::Q4_0:
                case DType::Q4_1:
                {
                    const bool affine = dtype == DType::Q4_1;
                    const uint8x16_t packed = vld1q_u8(encoded + (affine ? 4 : 2));
                    weight_low[row] = low_nibbles(packed);
                    weight_high[row] = high_nibbles(packed);
                    weight_scale[row] = load_f16(encoded);
                    weight_additive[row] = affine ? load_f16(encoded + 2) : 0.0f;
                    weight_bias[row] = affine ? 0.0f : 8.0f;
                    break;
                }
                case DType::Q5_0:
                case DType::Q5_1:
                {
                    const bool affine = dtype == DType::Q5_1;
                    const uint8_t* high = encoded + (affine ? 4 : 2);
                    const uint8x16_t packed = vld1q_u8(encoded + (affine ? 8 : 6));
                    weight_low[row] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(packed, vdupq_n_u8(0x0f)),
                                                                    q5_high_bits(high, 0)));
                    weight_high[row] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(packed, 4),
                                                                     q5_high_bits(high, 1)));
                    weight_scale[row] = load_f16(encoded);
                    weight_additive[row] = affine ? load_f16(encoded + 2) : 0.0f;
                    weight_bias[row] = affine ? 0.0f : 16.0f;
                    break;
                }
                case DType::IQ4NL:
                {
                    const uint8x16_t packed = vld1q_u8(encoded + 2);
                    weight_low[row] = vqtbl1q_s8(table, vandq_u8(packed, vdupq_n_u8(0x0f)));
                    weight_high[row] = vqtbl1q_s8(table, vshrq_n_u8(packed, 4));
                    weight_scale[row] = load_f16(encoded);
                    weight_additive[row] = 0.0f;
                    weight_bias[row] = 0.0f;
                    break;
                }
                default:
                    weight_low[row] = vdupq_n_s8(0);
                    weight_high[row] = vdupq_n_s8(0);
                    weight_scale[row] = 0.0f;
                    weight_additive[row] = 0.0f;
                    weight_bias[row] = 0.0f;
                    break;
                }
            }
            for (std::size_t first_token = 0; first_token < token_count; first_token += 4)
            {
                const uint32_t token_count_tile = static_cast<uint32_t>(
                    std::min<std::size_t>(4, token_count - first_token));
                if (token_count_tile == 4)
                {
                    for (uint32_t pair = 0; pair < 2; ++pair)
                    {
                        const uint32_t token0 = pair * 2;
                        const uint32_t token1 = token0 + 1;
                        const int8_t* codes0 = activation_codes
                                               + (first_token + token0) * static_cast<std::size_t>(columns)
                                               + static_cast<std::size_t>(block) * 32;
                        const int8_t* codes1 = activation_codes
                                               + (first_token + token1) * static_cast<std::size_t>(columns)
                                               + static_cast<std::size_t>(block) * 32;
                        const int8x16_t activation0_low = vld1q_s8(codes0);
                        const int8x16_t activation0_high = vld1q_s8(codes0 + 16);
                        const int8x16_t activation1_low = vld1q_s8(codes1);
                        const int8x16_t activation1_high = vld1q_s8(codes1 + 16);
                        const float activation0_scale = activation_scales[(first_token + token0) * block_count + block];
                        const float activation1_scale = activation_scales[(first_token + token1) * block_count + block];
                        const float activation0_sum = static_cast<float>(activation_sums[(first_token + token0) * block_count + block]);
                        const float activation1_sum = static_cast<float>(activation_sums[(first_token + token1) * block_count + block]);
                        for (uint32_t group = 0; group < 2; ++group)
                        {
                            const float32x4_t scales = vld1q_f32(weight_scale + group * 4);
                            const float32x4_t correction = vsubq_f32(vld1q_f32(weight_additive + group * 4),
                                                                     vmulq_f32(scales, vld1q_f32(weight_bias + group * 4)));
                            int32x4_t dot0[4];
                            int32x4_t dot1[4];
                            for (uint32_t row = 0; row < 4; ++row)
                            {
                                const uint32_t index = group * 4 + row;
                                dot0[row] = vdotq_s32(vdupq_n_s32(0), weight_low[index], activation0_low);
                                dot1[row] = vdotq_s32(vdupq_n_s32(0), weight_low[index], activation1_low);
                                dot0[row] = vdotq_s32(dot0[row], weight_high[index], activation0_high);
                                dot1[row] = vdotq_s32(dot1[row], weight_high[index], activation1_high);
                            }
                            const int32x4_t sums0 = vpaddq_s32(vpaddq_s32(dot0[0], dot0[1]),
                                                               vpaddq_s32(dot0[2], dot0[3]));
                            const int32x4_t sums1 = vpaddq_s32(vpaddq_s32(dot1[0], dot1[1]),
                                                               vpaddq_s32(dot1[2], dot1[3]));
                            float32x4_t value0 = vmulq_f32(vcvtq_f32_s32(sums0),
                                                           vmulq_n_f32(scales, activation0_scale));
                            float32x4_t value1 = vmulq_f32(vcvtq_f32_s32(sums1),
                                                           vmulq_n_f32(scales, activation1_scale));
                            value0 = vaddq_f32(value0, vmulq_n_f32(correction,
                                                                  activation0_scale * activation0_sum));
                            value1 = vaddq_f32(value1, vmulq_n_f32(correction,
                                                                  activation1_scale * activation1_sum));
                            float* destination0 = output + (first_token + token0) * output_stride + first_row + group * 4;
                            float* destination1 = output + (first_token + token1) * output_stride + first_row + group * 4;
                            if (block == 0)
                            {
                                vst1q_f32(destination0, value0);
                                vst1q_f32(destination1, value1);
                            }
                            else
                            {
                                vst1q_f32(destination0, vaddq_f32(vld1q_f32(destination0), value0));
                                vst1q_f32(destination1, vaddq_f32(vld1q_f32(destination1), value1));
                            }
                        }
                    }
                }
                else
                {
                    int8x16_t activation_low[4];
                    int8x16_t activation_high[4];
                    float activation_scale[4];
                    int32_t activation_sum[4];
                    for (uint32_t token = 0; token < token_count_tile; ++token)
                    {
                        const int8_t* codes = activation_codes
                                              + (first_token + token) * static_cast<std::size_t>(columns)
                                              + static_cast<std::size_t>(block) * 32;
                        activation_low[token] = vld1q_s8(codes);
                        activation_high[token] = vld1q_s8(codes + 16);
                        activation_scale[token] = activation_scales[(first_token + token) * block_count + block];
                        activation_sum[token] = activation_sums[(first_token + token) * block_count + block];
                    }
                    for (uint32_t token = 0; token < token_count_tile; ++token)
                        for (uint32_t row = 0; row < 8; ++row)
                        {
                            int32x4_t dot = vdotq_s32(vdupq_n_s32(0), weight_low[row], activation_low[token]);
                            dot = vdotq_s32(dot, weight_high[row], activation_high[token]);
                            const float value = weight_scale[row] * activation_scale[token]
                                                    * static_cast<float>(vaddvq_s32(dot))
                                                + (weight_additive[row] - weight_scale[row] * weight_bias[row])
                                                      * activation_scale[token] * static_cast<float>(activation_sum[token]);
                            float& destination = output[(first_token + token) * output_stride + first_row + row];
                            if (block == 0)
                                destination = value;
                            else
                                destination += value;
                        }
                }
            }
        }
    }
}

NCNN_MOE_ARM_DOTPROD void neon_qnk_prefill_q8(DType dtype,
                                                    const uint8_t* weights,
                                                    std::size_t rows,
                                                    uint32_t columns,
                                                    const int8_t* activation_codes,
                                                    const float* activation_scales,
                                                    const int32_t* activation_sums,
                                                    std::size_t token_count,
                                                    float* output,
                                                    std::size_t output_stride) noexcept
{
    if (!weights || !activation_codes || !activation_scales || !activation_sums || !output
        || rows == 0 || token_count == 0 || columns == 0 || columns % 32 != 0)
        return;
    const int8x16_t table = vld1q_s8(kIq4NlTable);
    const uint32_t block_count = columns / 32;
    const std::size_t block_bytes = qnk_block_bytes(dtype);
    const std::size_t row_tiles = (rows + 3) / 4;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        row_tiles, static_cast<std::size_t>(cpu_linear_team_size(operation_count, dtype))));
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
    for (int64_t tile_index = 0; tile_index < static_cast<int64_t>(row_tiles); ++tile_index)
    {
        const std::size_t first_row = static_cast<std::size_t>(tile_index) * 4;
        const uint32_t row_count = static_cast<uint32_t>(std::min<std::size_t>(4, rows - first_row));
        for (uint32_t block = 0; block < block_count; ++block)
        {
            int8x16_t weight_low[4];
            int8x16_t weight_high[4];
            float weight_scale[4];
            float weight_additive[4];
            float weight_bias[4];
            for (uint32_t row = 0; row < row_count; ++row)
            {
                const uint8_t* encoded = weights
                                         + ((first_row + row) * static_cast<std::size_t>(block_count) + block) * block_bytes;
                switch (dtype)
                {
                case DType::Q4_0:
                case DType::Q4_1:
                {
                    const bool affine = dtype == DType::Q4_1;
                    const uint8x16_t packed = vld1q_u8(encoded + (affine ? 4 : 2));
                    weight_low[row] = low_nibbles(packed);
                    weight_high[row] = high_nibbles(packed);
                    weight_scale[row] = load_f16(encoded);
                    weight_additive[row] = affine ? load_f16(encoded + 2) : 0.0f;
                    weight_bias[row] = affine ? 0.0f : 8.0f;
                    break;
                }
                case DType::Q5_0:
                case DType::Q5_1:
                {
                    const bool affine = dtype == DType::Q5_1;
                    const uint8_t* high = encoded + (affine ? 4 : 2);
                    const uint8x16_t packed = vld1q_u8(encoded + (affine ? 8 : 6));
                    weight_low[row] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(packed, vdupq_n_u8(0x0f)),
                                                                    q5_high_bits(high, 0)));
                    weight_high[row] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(packed, 4),
                                                                     q5_high_bits(high, 1)));
                    weight_scale[row] = load_f16(encoded);
                    weight_additive[row] = affine ? load_f16(encoded + 2) : 0.0f;
                    weight_bias[row] = affine ? 0.0f : 16.0f;
                    break;
                }
                case DType::IQ4NL:
                {
                    const uint8x16_t packed = vld1q_u8(encoded + 2);
                    weight_low[row] = vqtbl1q_s8(table, vandq_u8(packed, vdupq_n_u8(0x0f)));
                    weight_high[row] = vqtbl1q_s8(table, vshrq_n_u8(packed, 4));
                    weight_scale[row] = load_f16(encoded);
                    weight_additive[row] = 0.0f;
                    weight_bias[row] = 0.0f;
                    break;
                }
                default:
                    weight_low[row] = vdupq_n_s8(0);
                    weight_high[row] = vdupq_n_s8(0);
                    weight_scale[row] = 0.0f;
                    weight_additive[row] = 0.0f;
                    weight_bias[row] = 0.0f;
                    break;
                }
            }
            for (std::size_t first_token = 0; first_token < token_count; first_token += 4)
            {
                const uint32_t token_count_tile = static_cast<uint32_t>(
                    std::min<std::size_t>(4, token_count - first_token));
                int8x16_t activation_low[4];
                int8x16_t activation_high[4];
                float activation_scale[4];
                int32_t activation_sum[4];
                for (uint32_t token = 0; token < token_count_tile; ++token)
                {
                    const int8_t* codes = activation_codes
                                          + (first_token + token) * static_cast<std::size_t>(columns)
                                          + static_cast<std::size_t>(block) * 32;
                    activation_low[token] = vld1q_s8(codes);
                    activation_high[token] = vld1q_s8(codes + 16);
                    activation_scale[token] = activation_scales[(first_token + token) * block_count + block];
                    activation_sum[token] = activation_sums[(first_token + token) * block_count + block];
                }
                if (row_count == 4 && token_count_tile == 4)
                {
                    const float32x4_t scales = vld1q_f32(weight_scale);
                    const float32x4_t correction = vsubq_f32(vld1q_f32(weight_additive),
                                                             vmulq_f32(scales, vld1q_f32(weight_bias)));
                    for (uint32_t token = 0; token < 4; ++token)
                    {
                        int32x4_t dot[4];
                        for (uint32_t row = 0; row < 4; ++row)
                        {
                            dot[row] = vdotq_s32(vdupq_n_s32(0), weight_low[row], activation_low[token]);
                            dot[row] = vdotq_s32(dot[row], weight_high[row], activation_high[token]);
                        }
                        const int32x4_t sums = vpaddq_s32(vpaddq_s32(dot[0], dot[1]),
                                                          vpaddq_s32(dot[2], dot[3]));
                        float32x4_t value = vmulq_f32(vcvtq_f32_s32(sums),
                                                       vmulq_n_f32(scales, activation_scale[token]));
                        value = vaddq_f32(value,
                                          vmulq_n_f32(correction,
                                                      activation_scale[token]
                                                          * static_cast<float>(activation_sum[token])));
                        float* destination = output + (first_token + token) * output_stride + first_row;
                        if (block == 0)
                            vst1q_f32(destination, value);
                        else
                            vst1q_f32(destination, vaddq_f32(vld1q_f32(destination), value));
                    }
                }
                else
                {
                    for (uint32_t token = 0; token < token_count_tile; ++token)
                        for (uint32_t row = 0; row < row_count; ++row)
                        {
                            int32x4_t dot = vdotq_s32(vdupq_n_s32(0), weight_low[row], activation_low[token]);
                            dot = vdotq_s32(dot, weight_high[row], activation_high[token]);
                            const float value = weight_scale[row] * activation_scale[token]
                                                    * static_cast<float>(vaddvq_s32(dot))
                                                + (weight_additive[row] - weight_scale[row] * weight_bias[row])
                                                      * activation_scale[token] * static_cast<float>(activation_sum[token]);
                            float& destination = output[(first_token + token) * output_stride + first_row + row];
                            if (block == 0)
                                destination = value;
                            else
                                destination += value;
                        }
                }
            }
        }
    }
}
#else
float neon_qnk_q8_row(DType,
                      const uint8_t*,
                      std::size_t,
                      uint32_t,
                      const int8_t*,
                      const float*,
                      const int32_t*) noexcept
{
    return 0.0f;
}

void neon_qnk_prefill_q8_0(const uint8_t*,
                           std::size_t,
                           uint32_t,
                           const int8_t*,
                           const float*,
                           std::size_t,
                           float*,
                           std::size_t) noexcept
{
}

void neon_qnk_prefill_q8(DType,
                         const uint8_t*,
                         std::size_t,
                         uint32_t,
                         const int8_t*,
                         const float*,
                         const int32_t*,
                         std::size_t,
                         float*,
                         std::size_t) noexcept
{
}

void neon_qnk_prefill_q8_rows8(DType,
                               const uint8_t*,
                               std::size_t,
                               uint32_t,
                               const int8_t*,
                               const float*,
                               const int32_t*,
                               std::size_t,
                               float*,
                               std::size_t) noexcept
{
}
#endif

} // namespace moe
} // namespace ncnn

#endif // aarch64
