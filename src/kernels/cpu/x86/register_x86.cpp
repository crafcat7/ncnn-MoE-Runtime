#include "kernels/cpu/cpu_ops.h"

#include "kernels/cpu/x86/vector_x86_internal.h"
#include "kernels/cpu/x86/x86_internal.h"
#include "ncnn/moe/runtime.h"

namespace ncnn {
namespace moe {

void register_x86_cpu_ops(CpuKernelOps& ops, CpuBackendCaps& caps, uint64_t enabled_isa) noexcept
{
#if defined(NCNN_MOE_X86_SIMD) \
    && (defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
    if ((enabled_isa & CpuIsaX86Avx2Fma) != 0)
    {
        ops.float_dot = avx2_float_dot;
        ops.float_argmax = avx2_float_argmax;
        ops.float_gemm_4x4 = avx2_float_gemm_4x4;
        ops.float_gemm_4x8 = avx2_float_gemm_4x8;
        ops.bfloat16_gemm_4x8 = avx2_bfloat16_gemm_4x8;
        ops.float_exp_inplace = avx2_float_exp_inplace;
        ops.float_silu_inplace = avx2_float_silu_inplace;
        ops.int8_float_dot = avx2_int8_float_dot;
        ops.float_l2_scale_inplace = avx2_float_l2_scale_inplace;
        ops.float_rms_scale_inplace = avx2_float_rms_scale_inplace;
        ops.float_rms_norm = avx2_float_rms_norm;
        ops.bfloat16_rms_norm = avx2_bfloat16_rms_norm;
        ops.float_rope_inplace = avx2_float_rope_inplace;
        ops.float_scale_inplace = avx2_float_scale_inplace;
        ops.float_scaled_add = avx2_float_scaled_add;
        ops.float_scale_add = avx2_float_scale_add;
        ops.scale_add = avx2_float_scale_inplace_and_scaled_add;
        ops.scaled_acc = avx2_float_scaled_add_and_accumulate;
        ops.float_weighted_scale = avx2_float_weighted_scale;
        ops.bfloat16_weighted_scale = avx2_bfloat16_weighted_scale;
        ops.float_sigmoid_mul = avx2_float_sigmoid_mul;
        ops.float_silu_mul = avx2_float_silu_mul;
        ops.float_hc_pre_4 = avx2_float_hc_pre_4;
        ops.float_hc_post_4 = avx2_float_hc_post_4;

        ops.bfloat16_dot = avx2_bfloat16_dot;
        ops.float_to_bfloat16_array = avx2_float_to_bfloat16_array;
        ops.bfloat16_scaled_add = avx2_bfloat16_scaled_add;

        ops.float8_e4m3_block_dot = avx2_float8_e4m3_block_dot;
        ops.float8_e4m3_block_dot_rows4 = avx2_float8_e4m3_block_dot_rows4;

        ops.unpack_gguf_mxfp4_rows = avx2_unpack_gguf_mxfp4_rows;
        ops.mxfp4_dot = avx2_mxfp4_dot;
        ops.mxfp4_gemm_row = avx2_mxfp4_gemm_row;
        ops.mxfp4_matmul_rows2 = avx2_mxfp4_matmul_rows2;
        ops.mxfp4_matmul_row_pairs = avx2_mxfp4_matmul_row_pairs;
        ops.mxfp4_q8_dot = avx2_mxfp4_q8_dot;
        ops.mxfp4_q8_quantize = avx2_mxfp4_q8_quantize;
        ops.mxfp4_q8_gemm_row = avx2_mxfp4_q8_gemm_row;
        ops.mxfp4_q8_matmul_rows2 = avx2_mxfp4_q8_matmul_rows2;
        ops.mxfp4_q8_matmul_row_pairs = avx2_mxfp4_q8_matmul_row_pairs;
        ops.mxfp4_q8_packed_gemm = avx2_mxfp4_q8_packed_gemm;

        ops.qnk_dequantize_block_32 = avx2_qnk_deq32;
        ops.qnk_dot_block = avx2_qnk_dot_block;
        ops.qnk_dot_row = avx2_qnk_dot_row;
        ops.qnk_gemm = avx2_qnk_gemm;
        ops.qnk_q8k_quantize = avx2_qnk_q8k_quantize;
        ops.qnk_q8_quant = avx2_qnk_q8_quant;
        ops.qnk_q8_row = avx2_qnk_q8_row;

        caps.active_isa_flags |= CpuIsaX86Avx2Fma;
        caps.arch_name = "x86-avx2";
        caps.has_simd_exp = true;
        caps.bfloat16_dot_name = "x86-avx2-fp32-fma";
        caps.float8_name = "x86-avx2-fma";
        caps.float8_native = true;
        caps.mxfp4_kind = MxFp4KernelKind::X86Avx2;
        caps.mxfp4_name = "x86-avx2-fma";
        caps.mxfp4_decode_pair_group = 2;
        caps.mxfp4_q8_ready = true;
        caps.qnk_q8_supported = true;
    }

    if ((enabled_isa & CpuIsaX86Avx512) != 0)
    {
        ops.float_dot = avx512_float_dot;
        ops.float_argmax = avx512_float_argmax;
        ops.float_gemm_4x4 = avx512_float_gemm_4x4;
        ops.float_gemm_4x8 = avx512_float_gemm_4x8;
        ops.bfloat16_gemm_4x8 = avx512_bfloat16_gemm_4x8;
        ops.float_exp_inplace = avx512_float_exp_inplace;
        ops.float_silu_inplace = avx512_float_silu_inplace;
        ops.int8_float_dot = avx512_int8_float_dot;
        ops.float_l2_scale_inplace = avx512_float_l2_scale_inplace;
        ops.float_rms_scale_inplace = avx512_float_rms_scale_inplace;
        ops.float_rms_norm = avx512_float_rms_norm;
        ops.bfloat16_rms_norm = avx512_bfloat16_rms_norm;
        ops.float_rope_inplace = avx512_float_rope_inplace;
        ops.float_scale_inplace = avx512_float_scale_inplace;
        ops.float_scaled_add = avx512_float_scaled_add;
        ops.float_scale_add = avx512_float_scale_add;
        ops.scale_add = avx512_float_scale_inplace_and_scaled_add;
        ops.scaled_acc = avx512_float_scaled_add_and_accumulate;
        ops.gdn_step = avx512_gdn_step;
        ops.gdn_conv4 = avx512_gdn_conv4;
        ops.pv_f32 = avx512_pv_f32;
        ops.float_weighted_scale = avx512_float_weighted_scale;
        ops.bfloat16_weighted_scale = avx512_bfloat16_weighted_scale;
        ops.float_sigmoid_mul = avx512_float_sigmoid_mul;
        ops.float_silu_mul = avx512_float_silu_mul;
        ops.float_hc_pre_4 = avx512_float_hc_pre_4;
        ops.float_hc_post_4 = avx512_float_hc_post_4;

        ops.bfloat16_dot = avx512_bfloat16_dot;
        ops.bfloat16_scaled_add = avx512_bfloat16_scaled_add;
        ops.pv_bf16 = avx512_pv_bf16;

        ops.float8_e4m3_block_dot = avx512_float8_e4m3_block_dot;
        ops.float8_e4m3_block_dot_rows4 = avx512_float8_e4m3_block_dot_rows4;
        ops.quantize_float8_e4m3 = avx512_quantize_float8_e4m3;

        ops.mxfp4_dot = avx512_mxfp4_dot;
        ops.mxfp4_gemm_row = avx512_mxfp4_gemm_row;
        ops.mxfp4_matmul_rows2 = avx512_mxfp4_matmul_rows2;
        ops.mxfp4_matmul_row_pairs = avx512_mxfp4_matmul_row_pairs;
        ops.mxfp4_q8_quantize = avx512_mxfp4_q8_quantize;
        ops.mxfp4_q8_packed_gemm = avx512_mxfp4_q8_packed_gemm;

        ops.qnk_dot_block = avx512_qnk_dot_block;
        ops.qnk_gemm = avx512_qnk_gemm;
        ops.qnk_q8k_quantize = avx512_qnk_q8k_quantize;

        caps.active_isa_flags |= CpuIsaX86Avx512;
        caps.arch_name = "x86-avx512";
        caps.has_simd_exp = true;
        caps.bfloat16_dot_name = "x86-avx512-fp32-fma";
        caps.float8_name = "x86-avx512";
        caps.float8_native = true;
        caps.mxfp4_kind = MxFp4KernelKind::X86Avx512;
        caps.mxfp4_name = "x86-avx512";
        caps.mxfp4_decode_pair_group = 2;
        caps.mxfp4_q8_ready = true;
    }

    if ((enabled_isa & CpuIsaX86Avx512) != 0 && (enabled_isa & CpuIsaX86Avx512Bf16) != 0)
    {
        ops.float_to_bfloat16_array = avx512_float_to_bfloat16_array;
        ops.bfloat16_pair_dot = avx512_bfloat16_pair_dot;
        ops.bfloat16_single_token_linear = avx512_bfloat16_single_token_linear;
        ops.bfloat16_batched_linear = avx512_bfloat16_batched_linear;
        ops.bfloat16_float8_e4m3_block_dot = avx512_bfloat16_float8_e4m3_block_dot;
        ops.bfloat16_float8_e4m3_block_dot_rows = avx512_bfloat16_float8_e4m3_block_dot_rows;
        ops.bfloat16_float8_e4m3_block_dot_rows_batch = avx512_bfloat16_float8_e4m3_block_dot_rows_batch;
        caps.bf16_fast_dp = true;
        caps.active_isa_flags |= CpuIsaX86Avx512Bf16;
    }
#else
    (void)ops;
    (void)caps;
    (void)enabled_isa;
#endif
}

} // namespace moe
} // namespace ncnn
