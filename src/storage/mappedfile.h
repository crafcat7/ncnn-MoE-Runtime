#ifndef NCNN_MOE_MAPPEDFILE_H
#define NCNN_MOE_MAPPEDFILE_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace ncnn {
namespace moe {

class MappedFileRange final : public std::enable_shared_from_this<MappedFileRange>
{
public:
    [[nodiscard]] static Result<std::shared_ptr<MappedFileRange>> open(const std::filesystem::path& path, uint64_t offset, uint64_t byte_count);

    ~MappedFileRange();

    MappedFileRange(const MappedFileRange&) = delete;
    MappedFileRange& operator=(const MappedFileRange&) = delete;

    [[nodiscard]] const uint8_t* data() const noexcept
    {
        return data_ptr;
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return data_size;
    }

    [[nodiscard]] MxFp4ByteBuffer share_bytes();
    [[nodiscard]] std::shared_ptr<const uint8_t> share_data();
    void prefault() const noexcept;

private:
    MappedFileRange();

    void* view = nullptr;
    size_t view_size = 0;
    uint8_t* data_ptr = nullptr;
    size_t data_size = 0;
#if defined(_WIN32)
    void* file_handle;
    void* mapping_handle = nullptr;
#else
    int file_handle = -1;
#endif
};

// Ask the platform to stage a mapped slice without synchronously touching
// every page on the caller thread.  This is intentionally a best-effort hint:
// the mapping remains valid and demand paging is still the fallback.
void prefetch_mapped_memory(const void* data, size_t byte_count) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MAPPEDFILE_H
