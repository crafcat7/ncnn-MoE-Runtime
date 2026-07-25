#ifndef NCNN_MOE_TYPES_H
#define NCNN_MOE_TYPES_H

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

class NcnnLinearOperator;
class NcnnVulkanAttentionOperator;
class MappedFileRange;

[[nodiscard]] inline bool has_flag(uint32_t flags, uint32_t flag) noexcept
{
    return (flags & flag) != 0;
}

class MxFp4ByteBuffer
{
public:
    MxFp4ByteBuffer() = default;

    MxFp4ByteBuffer(
        std::initializer_list<uint8_t> values)
    {
        resize(values.size());
        size_t index = 0;
        for (uint8_t value : values)
            data_.get()[index++] = value;
    }

    MxFp4ByteBuffer(const MxFp4ByteBuffer& other)
    {
        assign(other.data(), other.size());
    }

    MxFp4ByteBuffer& operator=(
        const MxFp4ByteBuffer& other)
    {
        if (this != &other)
            assign(other.data(), other.size());
        return *this;
    }

    MxFp4ByteBuffer(MxFp4ByteBuffer&& other) noexcept
        : data_(std::move(other.data_)),
          size_(other.size_)
    {
        other.size_ = 0;
    }

    MxFp4ByteBuffer& operator=(
        MxFp4ByteBuffer&& other) noexcept
    {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }

    MxFp4ByteBuffer& operator=(
        std::initializer_list<uint8_t> values)
    {
        MxFp4ByteBuffer replacement(values);
        *this = std::move(replacement);
        return *this;
    }

    void resize(size_t count)
    {
        if (count == size_)
            return;
        std::shared_ptr<uint8_t> replacement;
        if (count != 0) {
            replacement = allocate(count);
            if (data_)
                std::memcpy(
                    replacement.get(),
                    data_.get(),
                    std::min(size_, count));
        }
        data_ = std::move(replacement);
        size_ = count;
    }

    void assign(const uint8_t* source, size_t count)
    {
        std::shared_ptr<uint8_t> replacement;
        if (count != 0) {
            replacement = allocate(count);
            std::memcpy(replacement.get(), source, count);
        }
        data_ = std::move(replacement);
        size_ = count;
    }

    [[nodiscard]] uint8_t* data() noexcept
    {
        return data_.get();
    }

    [[nodiscard]] const uint8_t* data() const noexcept
    {
        return data_.get();
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return size_;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return size_ == 0;
    }

    uint8_t& operator[](size_t index) noexcept
    {
        return data_.get()[index];
    }

    const uint8_t& operator[](size_t index) const noexcept
    {
        return data_.get()[index];
    }

    uint8_t& front() noexcept
    {
        return data_.get()[0];
    }

    const uint8_t& front() const noexcept
    {
        return data_.get()[0];
    }

    uint8_t& back() noexcept
    {
        return data_.get()[size_ - 1];
    }

    const uint8_t& back() const noexcept
    {
        return data_.get()[size_ - 1];
    }

private:
    friend class MappedFileRange;

    MxFp4ByteBuffer(
        std::shared_ptr<uint8_t> data,
        size_t size) noexcept
        : data_(std::move(data)),
          size_(size)
    {
    }

    [[nodiscard]] static std::shared_ptr<uint8_t> allocate(
        size_t count)
    {
        return std::shared_ptr<uint8_t>(
            new uint8_t[count],
            std::default_delete<uint8_t[]>());
    }

    std::shared_ptr<uint8_t> data_;
    size_t size_ = 0;
};

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

struct MxFp4FileStorage
{
    std::string blocks_path;
    uint64_t blocks_offset = 0;
    uint64_t blocks_bytes = 0;
    std::string scales_path;
    uint64_t scales_offset = 0;
    uint64_t scales_bytes = 0;
};

struct TensorData
{
    DType dtype = DType::Float32;
    std::vector<uint32_t> shape;
    std::vector<float> float32_data;
    std::vector<uint16_t> bfloat16_data;
    std::vector<int8_t> int8_data;
    std::vector<float> quantization_scales;
    MxFp4ByteBuffer mxfp4_blocks;
    MxFp4ByteBuffer mxfp4_scales;
    std::shared_ptr<const uint8_t> mapped_data;
    uint64_t mapped_byte_count = 0;
    std::shared_ptr<const MxFp4FileStorage> mxfp4_file_storage;
    std::shared_ptr<NcnnLinearOperator> linear_operator;

    [[nodiscard]] uint64_t element_count() const noexcept;
    [[nodiscard]] std::span<const float> float32_values() const noexcept;
    [[nodiscard]] std::span<const uint16_t> bfloat16_values() const noexcept;
    [[nodiscard]] std::span<const int8_t> int8_values() const noexcept;
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

inline std::span<const float> TensorData::float32_values() const noexcept
{
    if (!float32_data.empty())
        return float32_data;
    if (dtype != DType::Float32
        || !mapped_data
        || mapped_byte_count % sizeof(float) != 0
        || reinterpret_cast<uintptr_t>(mapped_data.get())
                   % alignof(float)
               != 0) {
        return {};
    }
    return {
        reinterpret_cast<const float*>(mapped_data.get()),
        static_cast<size_t>(mapped_byte_count / sizeof(float))};
}

inline std::span<const uint16_t> TensorData::bfloat16_values() const noexcept
{
    if (!bfloat16_data.empty())
        return bfloat16_data;
    if (dtype != DType::BFloat16
        || !mapped_data
        || mapped_byte_count % sizeof(uint16_t) != 0
        || reinterpret_cast<uintptr_t>(mapped_data.get())
                   % alignof(uint16_t)
               != 0) {
        return {};
    }
    return {
        reinterpret_cast<const uint16_t*>(mapped_data.get()),
        static_cast<size_t>(mapped_byte_count / sizeof(uint16_t))};
}

inline std::span<const int8_t> TensorData::int8_values() const noexcept
{
    if (!int8_data.empty())
        return int8_data;
    if (dtype != DType::Int8
        || !mapped_data) {
        return {};
    }
    return {
        reinterpret_cast<const int8_t*>(mapped_data.get()),
        static_cast<size_t>(mapped_byte_count)};
}

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_TYPES_H
