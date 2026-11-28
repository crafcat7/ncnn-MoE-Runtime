#include "kernels/cpu/cpu_ops.h"

namespace ncnn {
namespace moe {

void register_ncnn_cpu_ops(CpuKernelOps& ops,
                           CpuBackendCaps& caps,
                           uint64_t enabled_isa) noexcept
{
    (void)ops;
    (void)caps;
    (void)enabled_isa;
}

} // namespace moe
} // namespace ncnn
