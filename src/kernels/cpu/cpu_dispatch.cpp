#include "kernels/cpu/cpu_ops.h"

#include "engine/cpu.h"
#include "ncnn/moe/runtime.h"

#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>

namespace ncnn {
namespace moe {

struct CpuDispatchState
{
    CpuKernelOps ops;
    CpuBackendCaps caps;
};

static std::string read_cpu_isa_override() noexcept
{
#if defined(_MSC_VER)
    char* raw = nullptr;
    size_t len = 0;
    if (_dupenv_s(&raw, &len, "NCNN_MOE_CPU_ISA") != 0 || !raw)
        return {};
    std::unique_ptr<char, decltype(&std::free)> holder(raw, &std::free);
    return std::string(raw);
#else
    const char* raw = std::getenv("NCNN_MOE_CPU_ISA");
    return raw ? std::string(raw) : std::string();
#endif
}

static uint64_t resolve_enabled_isa(uint64_t detected) noexcept
{
    const std::string mode = read_cpu_isa_override();
    if (mode.empty() || mode == "auto")
        return detected;
    if (mode == "scalar" || mode == "none")
        return 0;
    if (mode == "avx2")
        return detected & CpuIsaX86Avx2Fma;
    if (mode == "avx512")
    {
        return detected & (CpuIsaX86Avx2Fma | CpuIsaX86Avx512 | CpuIsaX86Avx512Vnni | CpuIsaX86Avx512Bf16);
    }
    if (mode == "neon")
        return detected & (CpuIsaArmNeon | CpuIsaArmDotProd | CpuIsaArmI8mm | CpuIsaArmSve2);
    return detected;
}

static bool verify_cpu_ops_complete(const CpuKernelOps& ops) noexcept
{
    return ops.float_dot != nullptr
           && ops.float_argmax != nullptr
           && ops.float_gemm_4x4 != nullptr
           && ops.float_gemm_4x8 != nullptr
           && ops.bfloat16_gemm_4x8 != nullptr
           && ops.f32_to_f16 != nullptr
           && ops.bf16_to_f16 != nullptr
           && ops.f16_gemm_4x8 != nullptr
           && ops.f16_pv != nullptr
           && ops.f16_pv4 != nullptr
           && ops.f32_scale_max != nullptr
           && ops.f32_softmax != nullptr
           && ops.float_exp_inplace != nullptr
           && ops.float_silu_inplace != nullptr
           && ops.int8_float_dot != nullptr
           && ops.float_l2_scale_inplace != nullptr
           && ops.float_rms_scale_inplace != nullptr
           && ops.float_rms_norm != nullptr
           && ops.bfloat16_rms_norm != nullptr
           && ops.float_rope_inplace != nullptr
           && ops.float_scale_inplace != nullptr
           && ops.float_scaled_add != nullptr
           && ops.float_scale_add != nullptr
           && ops.scale_add != nullptr
           && ops.scaled_acc != nullptr
           && ops.gdn_step != nullptr
           && ops.gdn_conv4 != nullptr
           && ops.pv_f32 != nullptr
           && ops.float_weighted_scale != nullptr
           && ops.bfloat16_weighted_scale != nullptr
           && ops.float_sigmoid_mul != nullptr
           && ops.float_silu_mul != nullptr
           && ops.float_hc_pre_4 != nullptr
           && ops.float_hc_post_4 != nullptr
           && ops.bfloat16_dot != nullptr
           && ops.float_to_bfloat16_array != nullptr
           && ops.bfloat16_pair_dot != nullptr
           && ops.bfloat16_scaled_add != nullptr
           && ops.pv_bf16 != nullptr
           && ops.float8_e4m3_block_dot != nullptr
           && ops.float8_e4m3_block_dot_rows4 != nullptr
           && ops.quantize_float8_e4m3 != nullptr
           && ops.unpack_gguf_mxfp4_rows != nullptr
           && ops.mxfp4_dot != nullptr
           && ops.mxfp4_gemm_row != nullptr
           && ops.mxfp4_matmul_rows2 != nullptr
           && ops.mxfp4_matmul_row_pairs != nullptr
           && ops.mxfp4_q8_dot != nullptr
           && ops.mxfp4_q8_quantize != nullptr
           && ops.mxfp4_q8_gemm_row != nullptr
           && ops.mxfp4_q8_matmul_rows2 != nullptr
           && ops.mxfp4_q8_matmul_row_pairs != nullptr
           && ops.mxfp4_q8_packed_gemm != nullptr
           && ops.qnk_dequantize_block_32 != nullptr
           && ops.qnk_dot_block != nullptr
           && ops.qnk_dot_row != nullptr
           && ops.qnk_gemm != nullptr
           && ops.qnk_q8k_quantize != nullptr
           && ops.qnk_q8_quant != nullptr
           && ops.qnk_q8_row != nullptr;
}

static const CpuDispatchState& cpu_dispatch_state() noexcept
{
    static const CpuDispatchState state = []() {
        CpuDispatchState initialized;
        const uint64_t detected = cpu_isa_flags();
        const uint64_t enabled_isa = resolve_enabled_isa(detected);
        initialized.caps.detected_isa_flags = detected;

        register_scalar_cpu_ops(initialized.ops, initialized.caps);
        register_x86_cpu_ops(initialized.ops, initialized.caps, enabled_isa);
        register_arm_cpu_ops(initialized.ops, initialized.caps, enabled_isa);
        register_ncnn_cpu_ops(initialized.ops, initialized.caps, enabled_isa);

        assert(verify_cpu_ops_complete(initialized.ops));
        return initialized;
    }();
    return state;
}

const CpuKernelOps& cpu_ops() noexcept
{
    return cpu_dispatch_state().ops;
}

const CpuBackendCaps& cpu_backend_caps() noexcept
{
    return cpu_dispatch_state().caps;
}

} // namespace moe
} // namespace ncnn
