#ifndef NCNN_MOE_KERNELS_CPU_SCALAR_INTERNAL_H
#define NCNN_MOE_KERNELS_CPU_SCALAR_INTERNAL_H

#include "kernels/cpu/cpu_ops.h"

namespace ncnn {
namespace moe {

void scalar_mxfp4_q8_packed_gemm(const uint8_t* packed,
                                 uint32_t row_count,
                                 uint32_t block_count,
                                 uint32_t tile_rows,
                                 const int8_t* input,
                                 size_t input_stride,
                                 const float* input_scales,
                                 size_t scale_stride,
                                 size_t token_count,
                                 float* output,
                                 size_t output_stride) noexcept;

void register_scalar_bfloat16_ops(CpuKernelOps& ops, CpuBackendCaps& caps) noexcept;
void register_scalar_float8_ops(CpuKernelOps& ops, CpuBackendCaps& caps) noexcept;
void register_scalar_mxfp4_ops(CpuKernelOps& ops, CpuBackendCaps& caps) noexcept;
void register_scalar_qnk_ops(CpuKernelOps& ops, CpuBackendCaps& caps) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_KERNELS_CPU_SCALAR_INTERNAL_H
