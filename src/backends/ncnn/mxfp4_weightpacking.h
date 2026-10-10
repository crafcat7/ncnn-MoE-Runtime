#ifndef NCNN_MOE_MXFP4_WEIGHTPACKING_H
#define NCNN_MOE_MXFP4_WEIGHTPACKING_H

#include "kernels/ops.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace ncnn {
namespace moe {

// The caller validates segment sizes and the float alignment of the bias offset.
// Every byte is initialized, without clearing weights that are immediately copied.
inline void pack_mxfp4_weight_storage(std::span<uint8_t> storage,
                                      size_t packed_segment_size, size_t scales_segment_size,
                                      std::span<const uint8_t> blocks, std::span<const uint8_t> scales,
                                      std::span<const float> float_bias, std::span<const uint16_t> bfloat16_bias) noexcept
{
    std::memcpy(storage.data(), blocks.data(), blocks.size());
    std::memset(storage.data() + blocks.size(), 0, packed_segment_size - blocks.size());
    uint8_t* scale_values = storage.data() + packed_segment_size;
    std::memcpy(scale_values, scales.data(), scales.size());
    std::memset(scale_values + scales.size(), 0, scales_segment_size - scales.size());

    const size_t bias_offset = packed_segment_size + scales_segment_size;
    float* bias_values = reinterpret_cast<float*>(storage.data() + bias_offset);
    if (!float_bias.empty())
        std::copy(float_bias.begin(), float_bias.end(), bias_values);
    else
        for (size_t index = 0; index < bfloat16_bias.size(); ++index)
            bias_values[index] = bfloat16_to_float(bfloat16_bias[index]);
    const size_t bias_size = (float_bias.size() + bfloat16_bias.size()) * sizeof(float);
    std::memset(storage.data() + bias_offset + bias_size, 0, storage.size() - bias_offset - bias_size);
}

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MXFP4_WEIGHTPACKING_H
