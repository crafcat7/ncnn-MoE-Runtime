#ifndef NCNN_MOE_ACTIVATIONBUFFER_H
#define NCNN_MOE_ACTIVATIONBUFFER_H

#include "ncnn/moe/types.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
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

    // Copies are independent snapshots, including mapped GPU output storage.
    ActivationBuffer(const ActivationBuffer& other)
        : row_count(other.row_count), column_count(other.column_count), data(other.bytes().begin(), other.bytes().end()), type(other.type)
    {
    }
    ActivationBuffer& operator=(const ActivationBuffer& other)
    {
        if (this != &other)
        {
            const auto source = other.bytes();
            data.assign(source.begin(), source.end());
            external_owner.reset();
            external_data = nullptr;
            external_capacity = 0;
            row_count = other.row_count;
            column_count = other.column_count;
            type = other.type;
        }
        return *this;
    }
    ActivationBuffer(ActivationBuffer&& other) noexcept
    { swap(other); }
    ActivationBuffer& operator=(ActivationBuffer&& other) noexcept
    {
        if (this != &other)
        {
            ActivationBuffer moved(std::move(other));
            swap(moved);
        }
        return *this;
    }

    // The owner leases completed, contiguous host-visible storage. Recording
    // code must not publish this buffer until its completion checks succeed.
    void assign_external(size_t rows, uint32_t columns, DType dtype,
                         std::span<std::byte> storage, std::shared_ptr<void> owner)
    {
        const size_t element_size = element_size_for(dtype);
        if (element_size == 0 || !owner || !storage.data()
            || reinterpret_cast<uintptr_t>(storage.data()) % element_size != 0)
            throw std::invalid_argument("invalid external activation storage");
        if (columns != 0 && rows > data.max_size() / columns / element_size)
            throw std::length_error("external activation buffer is too large");
        if (rows * static_cast<size_t>(columns) * element_size > storage.size())
            throw std::length_error("external activation storage is too small");
        std::vector<std::byte>().swap(data);
        external_owner = std::move(owner);
        external_data = storage.data();
        external_capacity = storage.size();
        row_count = rows;
        column_count = columns;
        type = dtype;
    }

    void reset(size_t rows, uint32_t columns, bool clear)
    {
        const size_t element_size = element_size_for(type);
        if (element_size == 0)
            throw std::invalid_argument("unsupported activation storage type");
        if (columns != 0 && rows > data.max_size() / columns / element_size)
            throw std::length_error("activation buffer is too large");
        if (external_owner)
        {
            // Resizing a published view creates ordinary owned storage; it
            // never overwrites a still-leased device download destination.
            std::vector<std::byte> resized(rows * static_cast<size_t>(columns) * element_size);
            if (!clear && !resized.empty())
                std::memcpy(resized.data(), external_data, std::min(resized.size(), bytes().size()));
            data.swap(resized);
            external_owner.reset();
            external_data = nullptr;
            external_capacity = 0;
            row_count = rows;
            column_count = columns;
            return;
        }
        const size_t old_size = data.size();
        data.resize(rows * static_cast<size_t>(columns) * element_size);
        row_count = rows;
        column_count = columns;
        // resize initializes new bytes; only reused bytes need clearing.
        if (clear)
            std::fill_n(data.begin(), std::min(old_size, data.size()), std::byte{0});
    }

    void clear() noexcept
    {
        row_count = 0;
        column_count = 0;
        data.clear();
        external_owner.reset();
        external_data = nullptr;
        external_capacity = 0;
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
        external_owner.swap(other.external_owner);
        std::swap(external_data, other.external_data);
        std::swap(external_capacity, other.external_capacity);
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
        return reinterpret_cast<float*>(storage_data())
               + index * column_count;
    }

    [[nodiscard]] const float* row(size_t index) const
    {
        assert(type == DType::Float32);
        assert(index < row_count);
        return reinterpret_cast<const float*>(storage_data())
               + index * column_count;
    }

    [[nodiscard]] std::span<const std::byte> row_bytes(size_t index) const noexcept
    {
        assert(index < row_count);
        const size_t offset = index * static_cast<size_t>(column_count) * element_size();
        return {storage_data() + offset, static_cast<size_t>(column_count) * element_size()};
    }

    [[nodiscard]] std::span<std::byte> mutable_row_bytes(size_t index) noexcept
    {
        assert(index < row_count);
        const size_t offset = index * static_cast<size_t>(column_count) * element_size();
        return {storage_data() + offset, static_cast<size_t>(column_count) * element_size()};
    }

    [[nodiscard]] std::span<const float> values() const noexcept
    {
        assert(type == DType::Float32);
        return {reinterpret_cast<const float*>(storage_data()), row_count * column_count};
    }

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept
    {
        return {storage_data(), row_count * column_count * element_size()};
    }

    [[nodiscard]] std::span<std::byte> mutable_bytes() noexcept
    {
        return {storage_data(), row_count * column_count * element_size()};
    }

    [[nodiscard]] uint64_t allocated_bytes() const noexcept
    {
        return external_owner ? external_capacity : static_cast<uint64_t>(data.capacity());
    }

private:
    [[nodiscard]] std::byte* storage_data() noexcept
    { return external_owner ? external_data : data.data(); }
    [[nodiscard]] const std::byte* storage_data() const noexcept
    { return external_owner ? external_data : data.data(); }
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
        case DType::Q8_0:
        case DType::IQ4NL:
        case DType::Q5_1:
        case DType::Q4_0:
        case DType::Q4_1:
        case DType::Q5_0:
        case DType::IQ4XS:
        case DType::NvFp4:
        case DType::Int4:
            return 0;
        }
        return 0;
    }

    size_t row_count = 0;
    uint32_t column_count = 0;
    std::vector<std::byte> data;
    DType type = DType::Float32;
    std::shared_ptr<void> external_owner;
    std::byte* external_data = nullptr;
    size_t external_capacity = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_ACTIVATIONBUFFER_H
