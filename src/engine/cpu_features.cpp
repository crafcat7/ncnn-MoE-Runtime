#include "cpu_features.h"

#include "ncnn/moe/runtime.h"

#include <array>
#include <string>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif
#endif

namespace ncnn {
namespace moe {

static void append_name(std::string& names, const char* name)
{
    if (!names.empty())
        names += ',';
    names += name;
}

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
static uint32_t maximum_cpuid_leaf() noexcept
{
    int registers[4] = {};
    __cpuid(registers, 0);
    return static_cast<uint32_t>(registers[0]);
}

static std::array<uint32_t, 4> cpuid(uint32_t leaf, uint32_t subleaf) noexcept
{
    int registers[4] = {};
    __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
    return {static_cast<uint32_t>(registers[0]), static_cast<uint32_t>(registers[1]), static_cast<uint32_t>(registers[2]), static_cast<uint32_t>(registers[3])};
}

static uint64_t enabled_xstate() noexcept
{
    return _xgetbv(0);
}
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
static uint32_t maximum_cpuid_leaf() noexcept
{
    return __get_cpuid_max(0, nullptr);
}

static std::array<uint32_t, 4> cpuid(uint32_t leaf, uint32_t subleaf) noexcept
{
    std::array<uint32_t, 4> registers = {};
    __cpuid_count(leaf, subleaf, registers[0], registers[1], registers[2], registers[3]);
    return registers;
}

static uint64_t enabled_xstate() noexcept
{
    uint32_t low = 0;
    uint32_t high = 0;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return static_cast<uint64_t>(high) << 32 | low;
}
#endif

CpuIsaCapabilities detect_cpu_isa_capabilities() noexcept
{
    CpuIsaCapabilities result;

#if defined(__aarch64__) || defined(_M_ARM64)
    result.flags |= CpuIsaArmNeon;
    append_name(result.names, "neon");
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP_SVE)
    const unsigned long hardware = getauxval(AT_HWCAP);
    if ((hardware & HWCAP_SVE) != 0)
    {
        result.flags |= CpuIsaArmSve;
        append_name(result.names, "sve");
    }
#elif defined(__ARM_FEATURE_SVE)
    result.flags |= CpuIsaArmSve;
    append_name(result.names, "sve");
#endif
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_SVE2)
    const unsigned long hardware2 = getauxval(AT_HWCAP2);
    if ((hardware2 & HWCAP2_SVE2) != 0)
    {
        result.flags |= CpuIsaArmSve2;
        append_name(result.names, "sve2");
    }
#elif defined(__ARM_FEATURE_SVE2)
    result.flags |= CpuIsaArmSve2;
    append_name(result.names, "sve2");
#endif
#elif defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    const uint32_t maximum_leaf = maximum_cpuid_leaf();
    if (maximum_leaf >= 1)
    {
        const std::array<uint32_t, 4> leaf1 = cpuid(1, 0);
        const uint32_t ecx = leaf1[2];
        const bool osxsave = (ecx & (UINT32_C(1) << NCNN_MOE_CPUID_1_ECX_OSXSAVE_BIT)) != 0;
        const bool avx = (ecx & (UINT32_C(1) << NCNN_MOE_CPUID_1_ECX_AVX_BIT)) != 0;
        const bool fma = (ecx & (UINT32_C(1) << NCNN_MOE_CPUID_1_ECX_FMA_BIT)) != 0;
        const uint64_t xstate = osxsave ? enabled_xstate() : 0;
        const uint64_t avx_state_mask = (UINT64_C(1) << NCNN_MOE_XSTATE_XMM_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_YMM_BIT);
        const bool avx_state = avx && (xstate & avx_state_mask) == avx_state_mask;
        if (maximum_leaf >= 7 && avx_state)
        {
            const std::array<uint32_t, 4> leaf7 = cpuid(7, 0);
            const uint32_t ebx = leaf7[1];
            const uint32_t ecx7 = leaf7[2];
            const uint32_t edx = leaf7[3];
            const bool avx2 = (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX2_BIT)) != 0;
            if (avx2 && fma)
            {
                result.flags |= CpuIsaX86Avx2Fma;
                append_name(result.names, "avx2-fma");
            }

            const uint64_t avx512_state_mask = avx_state_mask | (UINT64_C(1) << NCNN_MOE_XSTATE_OPMASK_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_ZMM_HI256_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_HI16_ZMM_BIT);
            const bool avx512_state = (xstate & avx512_state_mask) == avx512_state_mask;
            const bool avx512 = avx512_state
                                && (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512F_BIT)) != 0
                                && (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512BW_BIT)) != 0
                                && (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512VL_BIT)) != 0;
            if (avx512)
            {
                result.flags |= CpuIsaX86Avx512;
                append_name(result.names, "avx512");
            }
            if (avx512 && (ecx7 & (UINT32_C(1) << NCNN_MOE_CPUID_7_ECX_AVX512VNNI_BIT)) != 0)
            {
                result.flags |= CpuIsaX86Avx512Vnni;
                append_name(result.names, "avx512-vnni");
            }

            if (maximum_leaf >= 7)
            {
                const std::array<uint32_t, 4> leaf71 = cpuid(7, 1);
                if (avx2 && (leaf71[0] & (UINT32_C(1) << NCNN_MOE_CPUID_7_1_EAX_AVXVNNI_BIT)) != 0)
                {
                    result.flags |= CpuIsaX86AvxVnni;
                    append_name(result.names, "avx-vnni");
                }
                if (avx512 && (leaf71[0] & (UINT32_C(1) << NCNN_MOE_CPUID_7_1_EAX_AVX512BF16_BIT)) != 0)
                {
                    result.flags |= CpuIsaX86Avx512Bf16;
                    append_name(result.names, "avx512-bf16");
                }
            }

            const uint64_t tile_state_mask = (UINT64_C(1) << NCNN_MOE_XSTATE_TILECFG_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_TILEDATA_BIT);
            const bool tile_state = (xstate & tile_state_mask) == tile_state_mask;
            const bool amx_tile = tile_state && (edx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EDX_AMX_TILE_BIT)) != 0;
            if (amx_tile)
            {
                result.flags |= CpuIsaX86AmxTile;
                append_name(result.names, "amx-tile");
            }
            if (amx_tile && (edx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EDX_AMX_INT8_BIT)) != 0)
            {
                result.flags |= CpuIsaX86AmxInt8;
                append_name(result.names, "amx-int8");
            }
            if (amx_tile && (edx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EDX_AMX_BF16_BIT)) != 0)
            {
                result.flags |= CpuIsaX86AmxBf16;
                append_name(result.names, "amx-bf16");
            }
        }
    }
#endif

    if (result.names.empty())
        result.names = "scalar";
    return result;
}

} // namespace moe
} // namespace ncnn
