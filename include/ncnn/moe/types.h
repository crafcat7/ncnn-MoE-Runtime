#ifndef NCNN_MOE_TYPES_H
#define NCNN_MOE_TYPES_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

class NcnnLinearOperator;

enum class DType
{
    Float32,
    Float16,
    BFloat16,
    Int8,
    MxFp4
};

enum class NormType
{
    None,
    RmsNorm
};

enum class RouterScoreFunction
{
    Softmax
};

enum class RouterNormalization
{
    None,
    SelectedExperts
};

enum class ExpertActivation
{
    Relu,
    Silu,
    Gelu,
    ClampedSilu,
    GptOssSwiGlu
};

enum class ExpertLayout
{
    UpDown,
    GateUpDown,
    InterleavedGateUpDown
};

enum class HybridMode
{
    CpuOnly,
    VulkanOnly,
    VulkanWithCpuPrefetch,
    HybridExperts,
    Auto
};

enum class LogitsOutputMode
{
    FullLogits,
    TopKCandidates,
    SampledToken
};

struct TensorData
{
    DType dtype = DType::Float32;
    std::vector<uint32_t> shape;
    std::vector<float> float32_data;
    std::vector<uint16_t> bfloat16_data;
    std::vector<int8_t> int8_data;
    std::vector<float> quantization_scales;
    std::vector<uint8_t> mxfp4_blocks;
    std::vector<uint8_t> mxfp4_scales;
    std::shared_ptr<NcnnLinearOperator> linear_operator;

    [[nodiscard]] uint64_t element_count() const noexcept;
};

struct WeightMapping
{
    std::unordered_map<std::string, TensorData> tensors;
};

inline uint64_t TensorData::element_count() const noexcept
{
    uint64_t count = 1;
    for (uint32_t dimension : shape)
        count *= dimension;
    return shape.empty() ? 0 : count;
}

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_TYPES_H
