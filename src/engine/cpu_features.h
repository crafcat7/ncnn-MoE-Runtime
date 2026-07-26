#ifndef NCNN_MOE_CPU_FEATURES_H
#define NCNN_MOE_CPU_FEATURES_H

#include <cstdint>
#include <string>

#define NCNN_MOE_CPUID_1_ECX_SSSE3_BIT        9
#define NCNN_MOE_CPUID_1_ECX_FMA_BIT          12
#define NCNN_MOE_CPUID_1_ECX_OSXSAVE_BIT      27
#define NCNN_MOE_CPUID_1_ECX_AVX_BIT          28
#define NCNN_MOE_CPUID_7_EBX_AVX2_BIT         5
#define NCNN_MOE_CPUID_7_EBX_AVX512F_BIT      16
#define NCNN_MOE_CPUID_7_EBX_AVX512BW_BIT     30
#define NCNN_MOE_CPUID_7_EBX_AVX512VL_BIT     31
#define NCNN_MOE_CPUID_7_ECX_AVX512VNNI_BIT   11
#define NCNN_MOE_CPUID_7_1_EAX_AVXVNNI_BIT    4
#define NCNN_MOE_CPUID_7_1_EAX_AVX512BF16_BIT 5
#define NCNN_MOE_CPUID_7_EDX_AMX_BF16_BIT     22
#define NCNN_MOE_CPUID_7_EDX_AMX_TILE_BIT     24
#define NCNN_MOE_CPUID_7_EDX_AMX_INT8_BIT     25
#define NCNN_MOE_XSTATE_XMM_BIT               1
#define NCNN_MOE_XSTATE_YMM_BIT               2
#define NCNN_MOE_XSTATE_OPMASK_BIT            5
#define NCNN_MOE_XSTATE_ZMM_HI256_BIT         6
#define NCNN_MOE_XSTATE_HI16_ZMM_BIT          7
#define NCNN_MOE_XSTATE_TILECFG_BIT           17
#define NCNN_MOE_XSTATE_TILEDATA_BIT          18

namespace ncnn {
namespace moe {

struct CpuIsaCapabilities
{
    uint64_t flags = 0;
    std::string names;
};

[[nodiscard]] CpuIsaCapabilities detect_cpu_isa_capabilities() noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_FEATURES_H
