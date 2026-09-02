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

static uint64_t detect_cpu_isa_flags() noexcept
{
    uint64_t flags = 0;

#if defined(__aarch64__) || defined(_M_ARM64)
    flags |= CpuIsaArmNeon;
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP_SVE)
    const unsigned long hardware = getauxval(AT_HWCAP);
    if ((hardware & HWCAP_SVE) != 0)
    {
        flags |= CpuIsaArmSve;
    }
#elif defined(__ARM_FEATURE_SVE)
    flags |= CpuIsaArmSve;
#endif
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_SVE2)
    const unsigned long hardware2 = getauxval(AT_HWCAP2);
    if ((hardware2 & HWCAP2_SVE2) != 0)
    {
        flags |= CpuIsaArmSve2;
    }
#elif defined(__ARM_FEATURE_SVE2)
    flags |= CpuIsaArmSve2;
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
                flags |= CpuIsaX86Avx2Fma;
            }

            const uint64_t avx512_state_mask = avx_state_mask | (UINT64_C(1) << NCNN_MOE_XSTATE_OPMASK_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_ZMM_HI256_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_HI16_ZMM_BIT);
            const bool avx512_state = (xstate & avx512_state_mask) == avx512_state_mask;
            const bool avx512 = avx512_state
                                && (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512F_BIT)) != 0
                                && (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512BW_BIT)) != 0
                                && (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512VL_BIT)) != 0;
            if (avx512)
            {
                flags |= CpuIsaX86Avx512;
            }
            if (avx512 && (ecx7 & (UINT32_C(1) << NCNN_MOE_CPUID_7_ECX_AVX512VNNI_BIT)) != 0)
            {
                flags |= CpuIsaX86Avx512Vnni;
            }

            if (maximum_leaf >= 7)
            {
                const std::array<uint32_t, 4> leaf71 = cpuid(7, 1);
                if (avx2 && (leaf71[0] & (UINT32_C(1) << NCNN_MOE_CPUID_7_1_EAX_AVXVNNI_BIT)) != 0)
                {
                    flags |= CpuIsaX86AvxVnni;
                }
                if (avx512 && (leaf71[0] & (UINT32_C(1) << NCNN_MOE_CPUID_7_1_EAX_AVX512BF16_BIT)) != 0)
                {
                    flags |= CpuIsaX86Avx512Bf16;
                }
            }

            const uint64_t tile_state_mask = (UINT64_C(1) << NCNN_MOE_XSTATE_TILECFG_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_TILEDATA_BIT);
            const bool tile_state = (xstate & tile_state_mask) == tile_state_mask;
            const bool amx_tile = tile_state && (edx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EDX_AMX_TILE_BIT)) != 0;
            if (amx_tile)
            {
                flags |= CpuIsaX86AmxTile;
            }
            if (amx_tile && (edx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EDX_AMX_INT8_BIT)) != 0)
            {
                flags |= CpuIsaX86AmxInt8;
            }
            if (amx_tile && (edx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EDX_AMX_BF16_BIT)) != 0)
            {
                flags |= CpuIsaX86AmxBf16;
            }
        }
    }
#endif

    return flags;
}

uint64_t cpu_isa_flags() noexcept
{
    static const uint64_t flags = detect_cpu_isa_flags();
    return flags;
}

std::string cpu_isa_names(uint64_t flags)
{
    std::string names;
    if (has_flag(flags, CpuIsaArmNeon))
        append_name(names, "neon");
    if (has_flag(flags, CpuIsaArmSve))
        append_name(names, "sve");
    if (has_flag(flags, CpuIsaArmSve2))
        append_name(names, "sve2");
    if (has_flag(flags, CpuIsaX86Avx2Fma))
        append_name(names, "avx2-fma");
    if (has_flag(flags, CpuIsaX86Avx512))
        append_name(names, "avx512");
    if (has_flag(flags, CpuIsaX86Avx512Vnni))
        append_name(names, "avx512-vnni");
    if (has_flag(flags, CpuIsaX86AvxVnni))
        append_name(names, "avx-vnni");
    if (has_flag(flags, CpuIsaX86Avx512Bf16))
        append_name(names, "avx512-bf16");
    if (has_flag(flags, CpuIsaX86AmxTile))
        append_name(names, "amx-tile");
    if (has_flag(flags, CpuIsaX86AmxInt8))
        append_name(names, "amx-int8");
    if (has_flag(flags, CpuIsaX86AmxBf16))
        append_name(names, "amx-bf16");
    return names.empty() ? "scalar" : names;
}

} // namespace moe
} // namespace ncnn
