#ifndef NCNN_MOE_INTERNAL_TENSOR_NAMES_H
#define NCNN_MOE_INTERNAL_TENSOR_NAMES_H

#include <cstdint>
#include <string>

namespace ncnn {
namespace moe {

inline std::string layer_prefix(uint32_t layer_id)
{
    return "layers." + std::to_string(layer_id) + ".";
}

inline std::string expert_prefix(uint32_t layer_id, uint32_t expert_id)
{
    return layer_prefix(layer_id) + "experts." + std::to_string(expert_id) + ".";
}

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_INTERNAL_TENSOR_NAMES_H
