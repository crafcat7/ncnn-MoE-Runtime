// blocked Q8_0 4x4 GEMM: token group outer, block inner, fp32
// accumulators across blocks. Compiled with -mcpu=apple-m3 -fno-lto.

#include "qnk.h"
#include "ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {
namespace {
inline float load_f16(const uint8_t* data) noexcept
{
    uint16_t bits = 0;
    std::memcpy(&bits, data, sizeof(bits));
    __fp16 half = 0.0f;
    std::memcpy(&half, &bits, sizeof(half));
    return static_cast<float>(half);
}
} // namespace

bool neon_qnk_i8mm_supported() noexcept
{
#if defined(__APPLE__)
    int value = 0;
    std::size_t size = sizeof(value);
    return sysctlbyname("hw.optional.arm.FEAT_I8MM", &value, &size, nullptr, 0) == 0
           && value != 0;
#elif defined(__linux__) && defined(HWCAP_I8MM)
    return (getauxval(AT_HWCAP2) & HWCAP_I8MM) != 0;
#else
    return false;
#endif
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
void neon_qnk_prefill_q8_0_gemm8(const uint8_t* weight_pack,
                                       std::size_t rows,
                                       uint32_t columns,
                                       const uint8_t* activation_pack,
                                       const float* activation_scales,
                                       std::size_t token_count,
                                       float* output,
                                       std::size_t output_stride) noexcept
{
    if (!weight_pack || !activation_pack || !activation_scales || !output
        || rows == 0 || rows % 4 != 0 || token_count == 0 || token_count % 8 != 0
        || columns == 0 || columns % 32 != 0)
        return;
    const uint32_t block_count = columns / 32;
    const std::size_t row_tiles = rows / 4;
    const std::size_t token_groups = token_count / 8;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const std::size_t job_count = token_groups * row_tiles;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        job_count, static_cast<std::size_t>(cpu_linear_team_size(operation_count, DType::Q8_0))));
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
    for (int64_t job = 0; job < static_cast<int64_t>(job_count); ++job)
    {
        const bool tile_first = team_size <= 10;
        const std::size_t tile = tile_first ? static_cast<std::size_t>(job) / token_groups
                                           : static_cast<std::size_t>(job) % row_tiles;
        const std::size_t group_pair = tile_first ? static_cast<std::size_t>(job) % token_groups
                                                 : static_cast<std::size_t>(job) / row_tiles;
        const std::size_t first_token = group_pair * 8;
        const std::size_t first_row = tile * 4;
        float32x4_t acc_f32[8];
        for (uint32_t index = 0; index < 8; ++index)
            acc_f32[index] = vdupq_n_f32(0.0f);
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t* b_ptr = weight_pack + (tile * block_count + block) * 144;
            int8x16_t b[4][2];
            for (uint32_t chunk = 0; chunk < 4; ++chunk)
            {
                b[chunk][0] = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32));
                b[chunk][1] = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32 + 16));
            }
            const float32x4_t b_d = {load_f16(b_ptr + 0), load_f16(b_ptr + 2), load_f16(b_ptr + 4), load_f16(b_ptr + 6)};
            for (uint32_t sub = 0; sub < 2; ++sub)
            {
                const std::size_t group = group_pair * 2 + sub;
                const uint8_t* a_ptr = activation_pack + (group * block_count + block) * 128;
                int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
                for (uint32_t chunk = 0; chunk < 4; ++chunk)
                {
                    const int8x16_t a01 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32));
                    const int8x16_t a23 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32 + 16));
                    acc[0] = vmmlaq_s32(acc[0], a01, b[chunk][0]);
                    acc[1] = vmmlaq_s32(acc[1], a01, b[chunk][1]);
                    acc[2] = vmmlaq_s32(acc[2], a23, b[chunk][0]);
                    acc[3] = vmmlaq_s32(acc[3], a23, b[chunk][1]);
                }
                const int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
                const int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
                const int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
                const int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));
                const float32x4_t a_d = {activation_scales[(first_token + sub * 4 + 0) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 1) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 2) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 3) * block_count + block]};
                acc_f32[sub * 4 + 0] = vfmaq_f32(acc_f32[sub * 4 + 0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
                acc_f32[sub * 4 + 1] = vfmaq_f32(acc_f32[sub * 4 + 1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
                acc_f32[sub * 4 + 2] = vfmaq_f32(acc_f32[sub * 4 + 2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
                acc_f32[sub * 4 + 3] = vfmaq_f32(acc_f32[sub * 4 + 3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));
            }
        }
        for (uint32_t row = 0; row < 8; ++row)
            vst1q_f32(output + (first_token + row) * output_stride + first_row, acc_f32[row]);
    }
}

void neon_qnk_prefill_q8_0_gemm16(const uint8_t* weight_pack,
                                       std::size_t rows,
                                       uint32_t columns,
                                       const uint8_t* activation_pack,
                                       const float* activation_scales,
                                       std::size_t token_count,
                                       float* output,
                                       std::size_t output_stride) noexcept
{
    if (!weight_pack || !activation_pack || !activation_scales || !output
        || rows == 0 || rows % 4 != 0 || token_count == 0 || token_count % 16 != 0
        || columns == 0 || columns % 32 != 0)
        return;
    const uint32_t block_count = columns / 32;
    const std::size_t row_tiles = rows / 4;
    const std::size_t token_groups = token_count / 16;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const std::size_t job_count = token_groups * row_tiles;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        job_count, static_cast<std::size_t>(cpu_linear_team_size(operation_count, DType::Q8_0))));
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
    for (int64_t job = 0; job < static_cast<int64_t>(job_count); ++job)
    {
        const bool tile_first = team_size <= 10;
        const std::size_t tile = tile_first ? static_cast<std::size_t>(job) / token_groups
                                           : static_cast<std::size_t>(job) % row_tiles;
        const std::size_t group_pair = tile_first ? static_cast<std::size_t>(job) % token_groups
                                                 : static_cast<std::size_t>(job) / row_tiles;
        const std::size_t first_token = group_pair * 16;
        const std::size_t first_row = tile * 4;
        float32x4_t acc_f32[16];
        for (uint32_t index = 0; index < 16; ++index)
            acc_f32[index] = vdupq_n_f32(0.0f);
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t* b_ptr = weight_pack + (tile * block_count + block) * 144;
            int8x16_t b[4][2];
            for (uint32_t chunk = 0; chunk < 4; ++chunk)
            {
                b[chunk][0] = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32));
                b[chunk][1] = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32 + 16));
            }
            const float32x4_t b_d = {load_f16(b_ptr + 0), load_f16(b_ptr + 2), load_f16(b_ptr + 4), load_f16(b_ptr + 6)};
            for (uint32_t sub = 0; sub < 4; ++sub)
            {
                const std::size_t group = group_pair * 4 + sub;
                const uint8_t* a_ptr = activation_pack + (group * block_count + block) * 128;
                int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
                for (uint32_t chunk = 0; chunk < 4; ++chunk)
                {
                    const int8x16_t a01 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32));
                    const int8x16_t a23 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32 + 16));
                    acc[0] = vmmlaq_s32(acc[0], a01, b[chunk][0]);
                    acc[1] = vmmlaq_s32(acc[1], a01, b[chunk][1]);
                    acc[2] = vmmlaq_s32(acc[2], a23, b[chunk][0]);
                    acc[3] = vmmlaq_s32(acc[3], a23, b[chunk][1]);
                }
                const int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
                const int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
                const int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
                const int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));
                const float32x4_t a_d = {activation_scales[(first_token + sub * 4 + 0) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 1) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 2) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 3) * block_count + block]};
                acc_f32[sub * 4 + 0] = vfmaq_f32(acc_f32[sub * 4 + 0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
                acc_f32[sub * 4 + 1] = vfmaq_f32(acc_f32[sub * 4 + 1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
                acc_f32[sub * 4 + 2] = vfmaq_f32(acc_f32[sub * 4 + 2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
                acc_f32[sub * 4 + 3] = vfmaq_f32(acc_f32[sub * 4 + 3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));
            }
        }
        for (uint32_t row = 0; row < 16; ++row)
            vst1q_f32(output + (first_token + row) * output_stride + first_row, acc_f32[row]);
    }
}

void neon_qnk_prefill_q8_0_fused16(const float* input,
                                   std::size_t tokens,
                                   uint32_t columns,
                                   const uint8_t* weight_pack,
                                   std::size_t rows,
                                   float* output,
                                   std::size_t output_stride) noexcept
{
    if (!input || !weight_pack || !output || tokens == 0 || tokens % 16 != 0
        || rows == 0 || rows % 4 != 0 || columns == 0 || columns % 32 != 0)
        return;
    const uint32_t block_count = columns / 32;
    const std::size_t row_tiles = rows / 4;
    const std::size_t token_groups = tokens / 16;
    const std::size_t quant_groups = tokens / 4;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * tokens;
    const std::size_t job_count = token_groups * row_tiles;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        job_count, static_cast<std::size_t>(cpu_linear_team_size(operation_count, DType::Q8_0))));
    std::vector<int8_t> activation_pack(tokens * static_cast<std::size_t>(columns));
    std::vector<float> activation_scales(tokens * static_cast<std::size_t>(block_count));
#pragma omp parallel num_threads(team_size)
    {
#pragma omp for schedule(static)
        for (int64_t group_index = 0; group_index < static_cast<int64_t>(quant_groups); ++group_index)
            neon_qnk_q8_quant_pack4_group(input, static_cast<std::size_t>(group_index), columns,
                                          activation_pack.data(), activation_scales.data());
#pragma omp for schedule(static)
        for (int64_t job = 0; job < static_cast<int64_t>(job_count); ++job)
        {
            const bool tile_first = team_size <= 10;
            const std::size_t tile = tile_first ? static_cast<std::size_t>(job) / token_groups
                                               : static_cast<std::size_t>(job) % row_tiles;
            const std::size_t group_pair = tile_first ? static_cast<std::size_t>(job) % token_groups
                                                     : static_cast<std::size_t>(job) / row_tiles;
            const std::size_t first_token = group_pair * 16;
            const std::size_t first_row = tile * 4;
            float32x4_t acc_f32[16];
            for (uint32_t index = 0; index < 16; ++index)
                acc_f32[index] = vdupq_n_f32(0.0f);
            for (uint32_t block = 0; block < block_count; ++block)
            {
                const uint8_t* b_ptr = weight_pack + (tile * block_count + block) * 144;
                int8x16_t b[4][2];
                for (uint32_t chunk = 0; chunk < 4; ++chunk)
                {
                    b[chunk][0] = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32));
                    b[chunk][1] = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32 + 16));
                }
                const float32x4_t b_d = {load_f16(b_ptr + 0), load_f16(b_ptr + 2),
                                         load_f16(b_ptr + 4), load_f16(b_ptr + 6)};
                for (uint32_t sub = 0; sub < 4; ++sub)
                {
                    const std::size_t group = group_pair * 4 + sub;
                    const uint8_t* a_ptr = reinterpret_cast<const uint8_t*>(activation_pack.data())
                                      + (group * block_count + block) * 128;
                    int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0),
                                        vdupq_n_s32(0), vdupq_n_s32(0)};
                    for (uint32_t chunk = 0; chunk < 4; ++chunk)
                    {
                        const int8x16_t a01 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32));
                        const int8x16_t a23 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32 + 16));
                        acc[0] = vmmlaq_s32(acc[0], a01, b[chunk][0]);
                        acc[1] = vmmlaq_s32(acc[1], a01, b[chunk][1]);
                        acc[2] = vmmlaq_s32(acc[2], a23, b[chunk][0]);
                        acc[3] = vmmlaq_s32(acc[3], a23, b[chunk][1]);
                    }
                    const int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
                    const int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
                    const int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
                    const int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));
                    const float32x4_t a_d = {activation_scales[(first_token + sub * 4 + 0) * block_count + block],
                                             activation_scales[(first_token + sub * 4 + 1) * block_count + block],
                                             activation_scales[(first_token + sub * 4 + 2) * block_count + block],
                                             activation_scales[(first_token + sub * 4 + 3) * block_count + block]};
                    acc_f32[sub * 4 + 0] = vfmaq_f32(acc_f32[sub * 4 + 0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
                    acc_f32[sub * 4 + 1] = vfmaq_f32(acc_f32[sub * 4 + 1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
                    acc_f32[sub * 4 + 2] = vfmaq_f32(acc_f32[sub * 4 + 2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
                    acc_f32[sub * 4 + 3] = vfmaq_f32(acc_f32[sub * 4 + 3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));
                }
            }
            for (uint32_t row = 0; row < 16; ++row)
                vst1q_f32(output + (first_token + row) * output_stride + first_row, acc_f32[row]);
        }
    }
}

void neon_qnk_prefill_q8_0_gemm12(const uint8_t* weight_pack,
                                       std::size_t rows,
                                       uint32_t columns,
                                       const uint8_t* activation_pack,
                                       const float* activation_scales,
                                       std::size_t token_count,
                                       float* output,
                                       std::size_t output_stride) noexcept
{
    if (!weight_pack || !activation_pack || !activation_scales || !output
        || rows == 0 || rows % 4 != 0 || token_count == 0 || token_count % 12 != 0
        || columns == 0 || columns % 32 != 0)
        return;
    const uint32_t block_count = columns / 32;
    const std::size_t row_tiles = rows / 4;
    const std::size_t token_groups = token_count / 12;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const std::size_t job_count = token_groups * row_tiles;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        job_count, static_cast<std::size_t>(cpu_linear_team_size(operation_count, DType::Q8_0))));
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
    for (int64_t job = 0; job < static_cast<int64_t>(job_count); ++job)
    {
        const bool tile_first = team_size <= 10;
        const std::size_t tile = tile_first ? static_cast<std::size_t>(job) / token_groups
                                           : static_cast<std::size_t>(job) % row_tiles;
        const std::size_t group_pair = tile_first ? static_cast<std::size_t>(job) % token_groups
                                                 : static_cast<std::size_t>(job) / row_tiles;
        const std::size_t first_token = group_pair * 12;
        const std::size_t first_row = tile * 4;
        float32x4_t acc_f32[12];
        for (uint32_t index = 0; index < 12; ++index)
            acc_f32[index] = vdupq_n_f32(0.0f);
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t* b_ptr = weight_pack + (tile * block_count + block) * 144;
            int8x16_t b[4][2];
            for (uint32_t chunk = 0; chunk < 4; ++chunk)
            {
                b[chunk][0] = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32));
                b[chunk][1] = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32 + 16));
            }
            const float32x4_t b_d = {load_f16(b_ptr + 0), load_f16(b_ptr + 2), load_f16(b_ptr + 4), load_f16(b_ptr + 6)};
            for (uint32_t sub = 0; sub < 3; ++sub)
            {
                const std::size_t group = group_pair * 3 + sub;
                const uint8_t* a_ptr = activation_pack + (group * block_count + block) * 128;
                int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
                for (uint32_t chunk = 0; chunk < 4; ++chunk)
                {
                    const int8x16_t a01 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32));
                    const int8x16_t a23 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32 + 16));
                    acc[0] = vmmlaq_s32(acc[0], a01, b[chunk][0]);
                    acc[1] = vmmlaq_s32(acc[1], a01, b[chunk][1]);
                    acc[2] = vmmlaq_s32(acc[2], a23, b[chunk][0]);
                    acc[3] = vmmlaq_s32(acc[3], a23, b[chunk][1]);
                }
                const int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
                const int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
                const int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
                const int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));
                const float32x4_t a_d = {activation_scales[(first_token + sub * 4 + 0) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 1) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 2) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 3) * block_count + block]};
                acc_f32[sub * 4 + 0] = vfmaq_f32(acc_f32[sub * 4 + 0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
                acc_f32[sub * 4 + 1] = vfmaq_f32(acc_f32[sub * 4 + 1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
                acc_f32[sub * 4 + 2] = vfmaq_f32(acc_f32[sub * 4 + 2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
                acc_f32[sub * 4 + 3] = vfmaq_f32(acc_f32[sub * 4 + 3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));
            }
        }
        for (uint32_t row = 0; row < 12; ++row)
            vst1q_f32(output + (first_token + row) * output_stride + first_row, acc_f32[row]);
    }
}

void neon_qnk_prefill_q8_0_gemm32(const uint8_t* weight_pack,
                                       std::size_t rows,
                                       uint32_t columns,
                                       const uint8_t* activation_pack,
                                       const float* activation_scales,
                                       std::size_t token_count,
                                       float* output,
                                       std::size_t output_stride) noexcept
{
    if (!weight_pack || !activation_pack || !activation_scales || !output
        || rows == 0 || rows % 4 != 0 || token_count == 0 || token_count % 32 != 0
        || columns == 0 || columns % 32 != 0)
        return;
    const uint32_t block_count = columns / 32;
    const std::size_t row_tiles = rows / 4;
    const std::size_t token_groups = token_count / 32;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const std::size_t job_count = token_groups * row_tiles;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        job_count, static_cast<std::size_t>(cpu_linear_team_size(operation_count, DType::Q8_0))));
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
    for (int64_t job = 0; job < static_cast<int64_t>(job_count); ++job)
    {
        const bool tile_first = team_size <= 10;
        const std::size_t tile = tile_first ? static_cast<std::size_t>(job) / token_groups
                                           : static_cast<std::size_t>(job) % row_tiles;
        const std::size_t group_pair = tile_first ? static_cast<std::size_t>(job) % token_groups
                                                 : static_cast<std::size_t>(job) / row_tiles;
        const std::size_t first_token = group_pair * 32;
        const std::size_t first_row = tile * 4;
        float32x4_t acc_f32[32];
        for (uint32_t index = 0; index < 32; ++index)
            acc_f32[index] = vdupq_n_f32(0.0f);
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t* b_ptr = weight_pack + (tile * block_count + block) * 144;
            int8x16_t b[4][2];
            for (uint32_t chunk = 0; chunk < 4; ++chunk)
            {
                b[chunk][0] = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32));
                b[chunk][1] = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32 + 16));
            }
            const float32x4_t b_d = {load_f16(b_ptr + 0), load_f16(b_ptr + 2), load_f16(b_ptr + 4), load_f16(b_ptr + 6)};
            for (uint32_t sub = 0; sub < 8; ++sub)
            {
                const std::size_t group = group_pair * 8 + sub;
                const uint8_t* a_ptr = activation_pack + (group * block_count + block) * 128;
                int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
                for (uint32_t chunk = 0; chunk < 4; ++chunk)
                {
                    const int8x16_t a01 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32));
                    const int8x16_t a23 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32 + 16));
                    acc[0] = vmmlaq_s32(acc[0], a01, b[chunk][0]);
                    acc[1] = vmmlaq_s32(acc[1], a01, b[chunk][1]);
                    acc[2] = vmmlaq_s32(acc[2], a23, b[chunk][0]);
                    acc[3] = vmmlaq_s32(acc[3], a23, b[chunk][1]);
                }
                const int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
                const int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
                const int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
                const int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));
                const float32x4_t a_d = {activation_scales[(first_token + sub * 4 + 0) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 1) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 2) * block_count + block],
                                         activation_scales[(first_token + sub * 4 + 3) * block_count + block]};
                acc_f32[sub * 4 + 0] = vfmaq_f32(acc_f32[sub * 4 + 0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
                acc_f32[sub * 4 + 1] = vfmaq_f32(acc_f32[sub * 4 + 1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
                acc_f32[sub * 4 + 2] = vfmaq_f32(acc_f32[sub * 4 + 2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
                acc_f32[sub * 4 + 3] = vfmaq_f32(acc_f32[sub * 4 + 3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));
            }
        }
        for (uint32_t row = 0; row < 32; ++row)
            vst1q_f32(output + (first_token + row) * output_stride + first_row, acc_f32[row]);
    }
}

void neon_qnk_prefill_q8_0_gemm(const uint8_t* weight_pack,
                                      std::size_t rows,
                                      uint32_t columns,
                                      const uint8_t* activation_pack,
                                      const float* activation_scales,
                                      std::size_t token_count,
                                      float* output,
                                      std::size_t output_stride) noexcept
{
    if (!weight_pack || !activation_pack || !activation_scales || !output
        || rows == 0 || rows % 4 != 0 || token_count == 0 || token_count % 4 != 0
        || columns == 0 || columns % 32 != 0)
        return;
    const uint32_t block_count = columns / 32;
    const std::size_t row_tiles = rows / 4;
    const std::size_t token_groups = token_count / 4;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const std::size_t job_count = token_groups * row_tiles;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        job_count, static_cast<std::size_t>(cpu_linear_team_size(operation_count, DType::Q8_0))));
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
    for (int64_t job = 0; job < static_cast<int64_t>(job_count); ++job)
    {
        const bool tile_first = team_size <= 10;
        const std::size_t tile = tile_first ? static_cast<std::size_t>(job) / token_groups
                                           : static_cast<std::size_t>(job) % row_tiles;
        const std::size_t group = tile_first ? static_cast<std::size_t>(job) % token_groups
                                            : static_cast<std::size_t>(job) / row_tiles;
        const std::size_t first_token = group * 4;
        const std::size_t first_row = tile * 4;
        float32x4_t acc_f32[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t* a_ptr = activation_pack + (group * block_count + block) * 128;
            const uint8_t* b_ptr = weight_pack + (tile * block_count + block) * 144;
            int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
            for (uint32_t chunk = 0; chunk < 4; ++chunk)
            {
                const int8x16_t a01 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32));
                const int8x16_t a23 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32 + 16));
                const int8x16_t b01 = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32));
                const int8x16_t b23 = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 16 + chunk * 32 + 16));
                acc[0] = vmmlaq_s32(acc[0], a01, b01);
                acc[1] = vmmlaq_s32(acc[1], a01, b23);
                acc[2] = vmmlaq_s32(acc[2], a23, b01);
                acc[3] = vmmlaq_s32(acc[3], a23, b23);
            }
            const int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
            const int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
            const int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
            const int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));
            const float32x4_t a_d = {activation_scales[(first_token + 0) * block_count + block],
                                     activation_scales[(first_token + 1) * block_count + block],
                                     activation_scales[(first_token + 2) * block_count + block],
                                     activation_scales[(first_token + 3) * block_count + block]};
            const float32x4_t b_d = {load_f16(b_ptr + 0), load_f16(b_ptr + 2), load_f16(b_ptr + 4), load_f16(b_ptr + 6)};
            acc_f32[0] = vfmaq_f32(acc_f32[0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
            acc_f32[1] = vfmaq_f32(acc_f32[1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
            acc_f32[2] = vfmaq_f32(acc_f32[2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
            acc_f32[3] = vfmaq_f32(acc_f32[3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));
        }
        for (uint32_t row = 0; row < 4; ++row)
            vst1q_f32(output + (first_token + row) * output_stride + first_row, acc_f32[row]);
    }
}
void neon_qnk_prefill_q8_0_gemm136(const uint8_t* weight_pack,
                                      std::size_t rows,
                                      uint32_t columns,
                                      const uint8_t* activation_pack,
                                      const float* activation_scales,
                                      std::size_t token_count,
                                      float* output,
                                      std::size_t output_stride) noexcept
{
    if (!weight_pack || !activation_pack || !activation_scales || !output
        || rows == 0 || rows % 4 != 0 || token_count == 0 || token_count % 4 != 0
        || columns == 0 || columns % 32 != 0)
        return;
    (void)activation_scales;
    const uint32_t block_count = columns / 32;
    const std::size_t row_tiles = rows / 4;
    const std::size_t token_groups = token_count / 4;
    const uint64_t operation_count = static_cast<uint64_t>(rows) * columns * token_count;
    const std::size_t job_count = token_groups * row_tiles;
    const int team_size = static_cast<int>(std::min<std::size_t>(
        job_count, static_cast<std::size_t>(cpu_linear_team_size(operation_count, DType::Q8_0))));
#pragma omp parallel for schedule(static) num_threads(team_size) if (team_size > 1)
    for (int64_t job = 0; job < static_cast<int64_t>(job_count); ++job)
    {
        const bool tile_first = team_size <= 10;
        const std::size_t tile = tile_first ? static_cast<std::size_t>(job) / token_groups
                                           : static_cast<std::size_t>(job) % row_tiles;
        const std::size_t group = tile_first ? static_cast<std::size_t>(job) % token_groups
                                            : static_cast<std::size_t>(job) / row_tiles;
        const std::size_t first_token = group * 4;
        const std::size_t first_row = tile * 4;
        float32x4_t acc_f32[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t* a_ptr = activation_pack + (group * block_count + block) * 128;
            const uint8_t* b_ptr = weight_pack + (tile * block_count + block) * 136;
            int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
            for (uint32_t chunk = 0; chunk < 4; ++chunk)
            {
                const int8x16_t a01 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32));
                const int8x16_t a23 = vld1q_s8(reinterpret_cast<const int8_t*>(a_ptr + chunk * 32 + 16));
                const int8x16_t b01 = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 8 + chunk * 32));
                const int8x16_t b23 = vld1q_s8(reinterpret_cast<const int8_t*>(b_ptr + 8 + chunk * 32 + 16));
                acc[0] = vmmlaq_s32(acc[0], a01, b01);
                acc[1] = vmmlaq_s32(acc[1], a01, b23);
                acc[2] = vmmlaq_s32(acc[2], a23, b01);
                acc[3] = vmmlaq_s32(acc[3], a23, b23);
            }
            const int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
            const int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
            const int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
            const int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));
            const float32x4_t a_d = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(a_ptr)));
            const float32x4_t b_d = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(b_ptr)));
            acc_f32[0] = vfmaq_f32(acc_f32[0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
            acc_f32[1] = vfmaq_f32(acc_f32[1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
            acc_f32[2] = vfmaq_f32(acc_f32[2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
            acc_f32[3] = vfmaq_f32(acc_f32[3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));
        }
        for (uint32_t row = 0; row < 4; ++row)
            vst1q_f32(output + (first_token + row) * output_stride + first_row, acc_f32[row]);
    }
}
#else
void neon_qnk_prefill_q8_0_gemm136(const uint8_t*,
                                        std::size_t,
                                        uint32_t,
                                        const uint8_t*,
                                        const float*,
                                        std::size_t,
                                        float*,
                                        std::size_t) noexcept
{
}

void neon_qnk_prefill_q8_0_gemm(const uint8_t*,
                                      std::size_t,
                                      uint32_t,
                                      const uint8_t*,
                                      const float*,
                                      std::size_t,
                                      float*,
                                      std::size_t) noexcept
{
}

void neon_qnk_prefill_q8_0_gemm8(const uint8_t*,
                                       std::size_t,
                                       uint32_t,
                                       const uint8_t*,
                                       const float*,
                                       std::size_t,
                                       float*,
                                       std::size_t) noexcept
{
}

void neon_qnk_prefill_q8_0_gemm16(const uint8_t*,
                                        std::size_t,
                                        uint32_t,
                                        const uint8_t*,
                                        const float*,
                                        std::size_t,
                                        float*,
                                        std::size_t) noexcept
{
}

void neon_qnk_prefill_q8_0_gemm12(const uint8_t*,
                                        std::size_t,
                                        uint32_t,
                                        const uint8_t*,
                                        const float*,
                                        std::size_t,
                                        float*,
                                        std::size_t) noexcept
{
}

void neon_qnk_prefill_q8_0_gemm32(const uint8_t*,
                                        std::size_t,
                                        uint32_t,
                                        const uint8_t*,
                                        const float*,
                                        std::size_t,
                                        float*,
                                        std::size_t) noexcept
{
}
#endif

} // namespace moe
} // namespace ncnn

#else
namespace ncnn {
namespace moe {
bool neon_qnk_i8mm_supported() noexcept { return false; }
void neon_qnk_prefill_q8_0_gemm136(const uint8_t*,
                                         std::size_t,
                                         uint32_t,
                                         const uint8_t*,
                                         const float*,
                                         std::size_t,
                                         float*,
                                         std::size_t) noexcept
{
}
void neon_qnk_prefill_q8_0_gemm(const uint8_t*,
                                      std::size_t,
                                      uint32_t,
                                      const uint8_t*,
                                      const float*,
                                      std::size_t,
                                      float*,
                                      std::size_t) noexcept
{
}

void neon_qnk_prefill_q8_0_gemm8(const uint8_t*,
                                       std::size_t,
                                       uint32_t,
                                       const uint8_t*,
                                       const float*,
                                       std::size_t,
                                       float*,
                                       std::size_t) noexcept
{
}

void neon_qnk_prefill_q8_0_gemm16(const uint8_t*,
                                        std::size_t,
                                        uint32_t,
                                        const uint8_t*,
                                        const float*,
                                        std::size_t,
                                        float*,
                                        std::size_t) noexcept
{
}

void neon_qnk_prefill_q8_0_gemm12(const uint8_t*,
                                        std::size_t,
                                        uint32_t,
                                        const uint8_t*,
                                        const float*,
                                        std::size_t,
                                        float*,
                                        std::size_t) noexcept
{
}

void neon_qnk_prefill_q8_0_gemm32(const uint8_t*,
                                        std::size_t,
                                        uint32_t,
                                        const uint8_t*,
                                        const float*,
                                        std::size_t,
                                        float*,
                                        std::size_t) noexcept
{
}
} // namespace moe
} // namespace ncnn
#endif
