#ifndef NCNN_MOE_ACTIVATION_H
#define NCNN_MOE_ACTIVATION_H

#include "ncnn/moe/types.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

// Contiguous host storage; dtype determines the byte layout.
class ActivationBuffer
{
public:
    ActivationBuffer() = default;
    ActivationBuffer(size_t rows, uint32_t columns, DType dtype = DType::Float32)
        : type(dtype)
    {
        reset(rows, columns, true);
    }

    ActivationBuffer(const ActivationBuffer&) = default;
    ActivationBuffer& operator=(const ActivationBuffer&) = default;
    ActivationBuffer(ActivationBuffer&&) noexcept = default;
    ActivationBuffer& operator=(ActivationBuffer&&) noexcept = default;

    void reset(size_t rows, uint32_t columns, bool clear)
    {
        const size_t element_size = element_size_for(type);
        if (element_size == 0)
            throw std::invalid_argument("unsupported activation storage type");
        if (columns != 0 && rows > data.max_size() / columns / element_size)
            throw std::length_error("activation buffer is too large");
        data.resize(rows * static_cast<size_t>(columns) * element_size);
        row_count = rows;
        column_count = columns;
        if (clear)
            std::fill(data.begin(), data.end(), std::byte{0});
    }

    void clear() noexcept
    {
        row_count = 0;
        column_count = 0;
        data.clear();
    }

    [[nodiscard]] DType dtype() const noexcept
    {
        return type;
    }

    [[nodiscard]] size_t element_size() const noexcept
    {
        return element_size_for(type);
    }

    void swap(ActivationBuffer& other) noexcept
    {
        std::swap(row_count, other.row_count);
        std::swap(column_count, other.column_count);
        data.swap(other.data);
        std::swap(type, other.type);
    }

    [[nodiscard]] size_t rows() const noexcept
    {
        return row_count;
    }

    [[nodiscard]] uint32_t columns() const noexcept
    {
        return column_count;
    }

    [[nodiscard]] float* row(size_t index)
    {
        assert(type == DType::Float32);
        assert(index < row_count);
        return reinterpret_cast<float*>(data.data())
               + index * column_count;
    }

    [[nodiscard]] const float* row(size_t index) const
    {
        assert(type == DType::Float32);
        assert(index < row_count);
        return reinterpret_cast<const float*>(data.data())
               + index * column_count;
    }

    [[nodiscard]] std::span<const std::byte> row_bytes(size_t index) const noexcept
    {
        assert(index < row_count);
        const size_t offset = index * static_cast<size_t>(column_count) * element_size();
        return {data.data() + offset, static_cast<size_t>(column_count) * element_size()};
    }

    [[nodiscard]] std::span<std::byte> mutable_row_bytes(size_t index) noexcept
    {
        assert(index < row_count);
        const size_t offset = index * static_cast<size_t>(column_count) * element_size();
        return {data.data() + offset, static_cast<size_t>(column_count) * element_size()};
    }

    [[nodiscard]] std::span<const float> values() const noexcept
    {
        assert(type == DType::Float32);
        return {reinterpret_cast<const float*>(data.data()), row_count * column_count};
    }

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept
    {
        return data;
    }

    [[nodiscard]] std::span<std::byte> mutable_bytes() noexcept
    {
        return data;
    }

    [[nodiscard]] uint64_t allocated_bytes() const noexcept
    {
        return static_cast<uint64_t>(data.capacity());
    }

private:
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

    size_t row_count = 0;
    uint32_t column_count = 0;
    std::vector<std::byte> data;
    DType type = DType::Float32;
};

using CpuBatch = ActivationBuffer;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_ACTIVATION_H
