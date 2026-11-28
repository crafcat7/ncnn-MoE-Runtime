#ifndef NCNN_MOE_KERNELS_CPU_CAPS_H
#define NCNN_MOE_KERNELS_CPU_CAPS_H

#include <cstdint>

namespace ncnn {
namespace moe {

enum class MxFp4KernelKind
{
    Scalar,
    ArmNeon,
    ArmSve2,
    X86Avx2,
    X86Avx512
};

struct CpuBackendCaps
{
    uint64_t detected_isa_flags = 0;
    uint64_t active_isa_flags = 0;
    const char* arch_name = "scalar";
    bool has_simd_exp = false;
    bool has_f16_vector = false;
    bool bf16_fast_dp = false;
    bool float8_native = false;
    bool qnk_q8_supported = false;
    bool qnk_i8mm_supported = false;
    const char* bfloat16_dot_name = "scalar-fp32";
    const char* float8_name = "scalar";
    MxFp4KernelKind mxfp4_kind = MxFp4KernelKind::Scalar;
    const char* mxfp4_name = "scalar";
    uint32_t mxfp4_decode_pair_group = 1;
    bool mxfp4_q8_ready = false;
};

[[nodiscard]] const CpuBackendCaps& cpu_backend_caps() noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_KERNELS_CPU_CAPS_H
