#ifndef NCNN_MOE_MOE_IR_H
#define NCNN_MOE_MOE_IR_H

#include "ncnn/moe/model_descriptor.h"

namespace ncnn {
namespace moe {

// MoeModelDescriptor is retained as the source-compatible name. MoeIR is the
// canonical model-neutral boundary emitted by adapters and consumed by the
// graph compiler.
using MoeIR = MoeModelDescriptor;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MOE_IR_H
