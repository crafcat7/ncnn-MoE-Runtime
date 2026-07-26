#ifndef NCNN_MOE_MAPPED_FILE_H
#define NCNN_MOE_MAPPED_FILE_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace ncnn {
namespace moe {

class MappedFile;

class MappedFileRange final : public std::enable_shared_from_this<MappedFileRange>
{
private:
    MappedFileRange() = default;

    std::shared_ptr<MappedFile> file_;
    void* view_ = nullptr;
    size_t view_size_ = 0;
    uint8_t* data_ = nullptr;
    size_t size_ = 0;

public:
    [[nodiscard]] static Result<std::shared_ptr<MappedFileRange>> open(const std::filesystem::path& path, uint64_t offset, uint64_t byte_count);

    ~MappedFileRange();

    MappedFileRange(const MappedFileRange&) = delete;
    MappedFileRange& operator=(const MappedFileRange&) = delete;

    [[nodiscard]] const uint8_t* data() const noexcept
    {
        return data_;
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return size_;
    }

    [[nodiscard]] MxFp4ByteBuffer share_bytes();
    [[nodiscard]] std::shared_ptr<const uint8_t> share_data();
    void prefault() const noexcept;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MAPPED_FILE_H
