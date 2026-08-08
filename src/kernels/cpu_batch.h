#ifndef NCNN_MOE_CPU_BATCH_H
#define NCNN_MOE_CPU_BATCH_H

#include "ncnn/moe/types.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

class ActivationBuffer
{
private:
    size_t rows_ = 0;
    uint32_t columns_ = 0;
    std::vector<std::byte> data_;
    DType dtype_ = DType::Float32;
    TensorLocation location_ = TensorLocation::Cpu;

    [[nodiscard]] static constexpr size_t element_size_for(DType dtype) noexcept
    {
        switch (dtype)
        {
        case DType::Float32:
        case DType::Int32:
            return sizeof(uint32_t);
        case DType::Int64:
            return sizeof(uint64_t);
        case DType::Float16:
        case DType::BFloat16:
            return sizeof(uint16_t);
        case DType::Float8E4M3:
        case DType::Int8:
            return sizeof(uint8_t);
        case DType::MxFp4:
        case DType::Q2K:
        case DType::Q3K:
        case DType::Q4K:
        case DType::Q5K:
        case DType::Q6K:
        case DType::Q8K:
            return 0;
        }
        return 0;
    }

public:
    ActivationBuffer() = default;
    ActivationBuffer(size_t rows, uint32_t columns, DType dtype = DType::Float32, TensorLocation location = TensorLocation::Cpu)
        : dtype_(dtype),
          location_(location)
    {
        reset(rows, columns, true);
    }

    ActivationBuffer(const ActivationBuffer&) = default;
    ActivationBuffer& operator=(const ActivationBuffer&) = default;
    ActivationBuffer(ActivationBuffer&&) noexcept = default;
    ActivationBuffer& operator=(ActivationBuffer&&) noexcept = default;

    void reset(size_t rows, uint32_t columns, bool clear)
    {
        const size_t element_size = element_size_for(dtype_);
        assert(element_size != 0);
        rows_ = rows;
        columns_ = columns;
        data_.resize(rows * static_cast<size_t>(columns) * element_size);
        if (clear)
            std::fill(data_.begin(), data_.end(), std::byte{0});
    }

    void clear() noexcept
    {
        rows_ = 0;
        columns_ = 0;
        data_.clear();
    }

    void set_type(DType dtype, TensorLocation location = TensorLocation::Cpu)
    {
        assert(element_size_for(dtype) != 0);
        dtype_ = dtype;
        location_ = location;
        if (rows_ != 0)
            reset(rows_, columns_, true);
    }

    [[nodiscard]] DType dtype() const noexcept
    {
        return dtype_;
    }

    [[nodiscard]] TensorLocation location() const noexcept
    {
        return location_;
    }

    [[nodiscard]] size_t element_size() const noexcept
    {
        return element_size_for(dtype_);
    }

    void swap(ActivationBuffer& other) noexcept
    {
        std::swap(rows_, other.rows_);
        std::swap(columns_, other.columns_);
        data_.swap(other.data_);
        std::swap(dtype_, other.dtype_);
        std::swap(location_, other.location_);
    }

    [[nodiscard]] size_t rows() const noexcept
    {
        return rows_;
    }

    [[nodiscard]] uint32_t columns() const noexcept
    {
        return columns_;
    }

    [[nodiscard]] float* row(size_t index)
    {
        assert(dtype_ == DType::Float32);
        assert(index < rows_);
        return reinterpret_cast<float*>(data_.data())
               + index * columns_;
    }

    [[nodiscard]] const float* row(size_t index) const
    {
        assert(dtype_ == DType::Float32);
        assert(index < rows_);
        return reinterpret_cast<const float*>(data_.data())
               + index * columns_;
    }

    [[nodiscard]] std::span<const std::byte> row_bytes(size_t index) const noexcept
    {
        assert(index < rows_);
        const size_t offset = index * static_cast<size_t>(columns_) * element_size();
        return {data_.data() + offset, static_cast<size_t>(columns_) * element_size()};
    }

    [[nodiscard]] std::span<std::byte> mutable_row_bytes(size_t index) noexcept
    {
        assert(index < rows_);
        const size_t offset = index * static_cast<size_t>(columns_) * element_size();
        return {data_.data() + offset, static_cast<size_t>(columns_) * element_size()};
    }

    [[nodiscard]] std::span<const float> values() const noexcept
    {
        assert(dtype_ == DType::Float32);
        return {reinterpret_cast<const float*>(data_.data()), rows_ * columns_};
    }

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept
    {
        return data_;
    }

    [[nodiscard]] std::span<std::byte> mutable_bytes() noexcept
    {
        return data_;
    }

    [[nodiscard]] uint64_t allocated_bytes() const noexcept
    {
        return static_cast<uint64_t>(data_.capacity());
    }
};

// CpuBatch remains the concrete FP32 host view used by CPU kernels.  GPU
// contracts use ActivationBuffer and must inspect its dtype instead of
// assuming an FP32 ABI.
class CpuBatch : public ActivationBuffer
{
public:
    using ActivationBuffer::ActivationBuffer;
    using ActivationBuffer::operator=;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_BATCH_H
