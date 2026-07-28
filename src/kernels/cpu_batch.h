#ifndef NCNN_MOE_CPU_BATCH_H
#define NCNN_MOE_CPU_BATCH_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

class CpuBatch
{
private:
    size_t rows_ = 0;
    uint32_t columns_ = 0;
    std::vector<float> data_;

public:
    CpuBatch() = default;
    CpuBatch(size_t rows, uint32_t columns)
    {
        reset(rows, columns, true);
    }

    void reset(size_t rows, uint32_t columns, bool clear)
    {
        rows_ = rows;
        columns_ = columns;
        data_.resize(rows * columns);
        if (clear)
            std::fill(data_.begin(), data_.end(), 0.0f);
    }

    void clear() noexcept
    {
        rows_ = 0;
        columns_ = 0;
        data_.clear();
    }

    void swap(CpuBatch& other) noexcept
    {
        std::swap(rows_, other.rows_);
        std::swap(columns_, other.columns_);
        data_.swap(other.data_);
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
        assert(index < rows_);
        return data_.data() + index * columns_;
    }

    [[nodiscard]] const float* row(size_t index) const
    {
        assert(index < rows_);
        return data_.data() + index * columns_;
    }

    [[nodiscard]] std::span<const float> values() const noexcept
    {
        return data_;
    }

    [[nodiscard]] uint64_t allocated_bytes() const noexcept
    {
        return static_cast<uint64_t>(data_.capacity()) * sizeof(float);
    }
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_BATCH_H
