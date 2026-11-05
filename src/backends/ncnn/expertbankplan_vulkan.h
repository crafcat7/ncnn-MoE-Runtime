#ifndef NCNN_MOE_EXPERTBANKPLAN_VULKAN_H
#define NCNN_MOE_EXPERTBANKPLAN_VULKAN_H

#include "ncnn/moe/types.h"

#include <cstdint>

namespace ncnn {
namespace moe {

// Device-independent layout metadata for one immutable Expert bank. Kept in a
// lightweight header so graph/model declarations do not pull in ncnn Vulkan.
struct ExpertBankPlan_vulkan
{
    DType dtype = DType::Float32;
    ExpertLayout layout = ExpertLayout::InterleavedGateUpDown;
    ExpertActivation activation = ExpertActivation::Silu;
    float activation_limit = 0.0f;
    uint32_t expert_count = 0;
    uint32_t input_columns = 0;
    uint32_t intermediate_columns = 0;
    uint32_t output_columns = 0;
    uint32_t device_index = 0;
    uint64_t storage_size = 0;
    uint64_t physical_size = 0;
    uint64_t alignment = 4;
    // Byte offsets and expert strides. Gate offsets are relative to each gate
    // slice; down_scales/down_bias are relative to each down slice, whose bank
    // begins at down_weights in the single immutable weight buffer.
    uint64_t gate_stride = 0;
    uint64_t down_stride = 0;
    uint64_t gate_scales = 0;
    uint64_t gate_bias = 0;
    uint64_t down_weights = 0;
    uint64_t down_scales = 0;
    uint64_t down_bias = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERTBANKPLAN_VULKAN_H
