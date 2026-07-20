#ifndef NCNN_MOE_CPU_BATCH_H
#define NCNN_MOE_CPU_BATCH_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

class CpuBatch
{
public:
    CpuBatch() = default;
    CpuBatch(size_t rows, uint32_t columns)
        : rows_(rows), columns_(columns), data_(rows * columns, 0.0f)
    {
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

private:
    size_t rows_ = 0;
    uint32_t columns_ = 0;
    std::vector<float> data_;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_BATCH_H
