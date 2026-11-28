#include "kernels/cpu/cpu_ops.h"

#include "kernels/cpu/arm/arm_internal.h"
#include "kernels/cpu/arm/vector_arm_internal.h"
#include "ncnn/moe/runtime.h"

namespace ncnn {
namespace moe {

void register_arm_cpu_ops(CpuKernelOps& ops, CpuBackendCaps& caps, uint64_t enabled_isa) noexcept
{
#if defined(__aarch64__) || defined(_M_ARM64)
    if ((enabled_isa & CpuIsaArmNeon) != 0)
    {
        ops.float_dot = neon_float_dot;
        ops.float_gemm_4x4 = neon_float_gemm;
        ops.float_gemm_4x8 = neon_float_gemm;
        ops.bfloat16_gemm_4x8 = neon_bfloat16_gemm;
        ops.float_rms_norm = neon_float_rms_norm;
        ops.bfloat16_rms_norm = neon_bfloat16_rms_norm;
        ops.float_rope_inplace = neon_float_rope_inplace;
        ops.f32_scale_max = neon_f32_scale_max;
        ops.f32_softmax = neon_f32_softmax;
        ops.float_exp_inplace = neon_float_exp_inplace;
        ops.float_sigmoid_mul = neon_float_sigmoid_mul;
        ops.float_silu_mul = neon_float_silu_mul;
        ops.float_silu_inplace = neon_float_silu_inplace;

        ops.bfloat16_dot = neon_bfloat16_dot;
        ops.float_to_bfloat16_array = neon_float_to_bfloat16_array;
        ops.bfloat16_scaled_add = neon_bfloat16_scaled_add;

        ops.mxfp4_dot = neon_mxfp4_dot;
        ops.mxfp4_gemm_row = neon_mxfp4_gemm_row;
        ops.mxfp4_matmul_rows2 = neon_mxfp4_matmul_rows2;
        ops.mxfp4_matmul_row_pairs = neon_mxfp4_matmul_row_pairs;

        ops.qnk_q8k_quantize = neon_qnk_q8k_quantize;

        caps.active_isa_flags |= CpuIsaArmNeon;
        caps.arch_name = "arm-neon";
        caps.has_simd_exp = true;
        caps.bfloat16_dot_name = "arm-neon-fp32-fma";
        caps.mxfp4_kind = MxFp4KernelKind::ArmNeon;
        caps.mxfp4_name = "arm-neon";
        caps.mxfp4_decode_pair_group = 1;

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        ops.f32_to_f16 = neon_f32_to_f16;
        ops.bf16_to_f16 = neon_bf16_to_f16;
        ops.f16_gemm_4x8 = neon_f16_gemm_4x8;
        ops.f16_pv = neon_f16_pv;
        ops.f16_pv4 = neon_f16_pv4;

        caps.has_f16_vector = true;
        caps.arch_name = "arm-neon-fp16";
#endif

        if ((enabled_isa & CpuIsaArmDotProd) != 0 && neon_qnk_q8_supported())
        {
            ops.qnk_q8_quant = neon_qnk_q8_quant;
            ops.qnk_q8_row = neon_qnk_q8_row;
            ops.qnk_q8_quant_pack4 = neon_qnk_q8_quant_pack4;
            ops.qnk_prefill_q8_0 = neon_qnk_prefill_q8_0;
            ops.qnk_prefill_q8_0_rows8 = neon_qnk_prefill_q8_0_rows8;
            ops.qnk_prefill_q8 = neon_qnk_prefill_q8;
            ops.qnk_prefill_q8_rows8 = neon_qnk_prefill_q8_rows8;
            caps.qnk_q8_supported = true;
            caps.active_isa_flags |= CpuIsaArmDotProd;

            if ((enabled_isa & CpuIsaArmI8mm) != 0 && neon_qnk_i8mm_supported())
            {
                ops.qnk_prefill_q8_0_gemm4 = neon_qnk_prefill_q8_0_gemm4;
                ops.qnk_prefill_q8_0_gemm8 = neon_qnk_prefill_q8_0_gemm8;
                ops.qnk_prefill_q8_0_gemm16 = neon_qnk_prefill_q8_0_gemm16;
                caps.qnk_i8mm_supported = true;
                caps.active_isa_flags |= CpuIsaArmI8mm;
            }
        }

#if defined(NCNN_MOE_ARM_SVE2_KERNEL)
        if ((enabled_isa & CpuIsaArmSve2) != 0)
        {
            ops.mxfp4_dot = sve2_mxfp4_dot;
            ops.mxfp4_matmul_rows2 = sve2_mxfp4_matmul_rows2;
            ops.mxfp4_matmul_row_pairs = sve2_mxfp4_matmul_row_pairs;
            caps.mxfp4_kind = MxFp4KernelKind::ArmSve2;
            caps.mxfp4_name = "arm-sve2";
            caps.active_isa_flags |= CpuIsaArmSve2;
        }
#endif
    }
#else
    (void)ops;
    (void)caps;
    (void)enabled_isa;
#endif
}

} // namespace moe
} // namespace ncnn
