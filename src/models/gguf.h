#ifndef NCNN_MOE_GGUF_H
#define NCNN_MOE_GGUF_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ncnn {
namespace moe {

enum class GgufMetadataType : uint32_t
{
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12
};

struct GgufMetadataValue
{
    using Array = std::vector<GgufMetadataValue>;
    GgufMetadataType type = GgufMetadataType::Uint8;
    GgufMetadataType array_type = GgufMetadataType::Uint8;
    std::variant<uint64_t, int64_t, double, bool, std::string, Array> value;

    [[nodiscard]] std::optional<uint64_t> unsigned_integer() const noexcept;
    [[nodiscard]] std::optional<int64_t> integer() const noexcept;
    [[nodiscard]] std::optional<double> number() const noexcept;
    [[nodiscard]] std::optional<bool> boolean() const noexcept;
    [[nodiscard]] const std::string* string() const noexcept;
    [[nodiscard]] const Array* array() const noexcept;
    bool operator==(const GgufMetadataValue&) const = default;
};

struct GgufTensorInfo
{
    std::filesystem::path path;
    std::string name;
    DType dtype = DType::Float32;
    uint32_t ggml_type = 0;
    // GGUF stores the contiguous dimension first; runtime shapes reverse it.
    std::vector<uint32_t> shape;
    uint64_t offset = 0;
    uint64_t size = 0;
};

class GgufArchive
{
public:
    // Accept one file (including any standard split shard), a GGUF directory,
    // or its parent containing one GGUF package in an immediate subdirectory.
    [[nodiscard]] static Result<GgufArchive> open(const std::filesystem::path& root_or_file);
    [[nodiscard]] const GgufTensorInfo* find(const std::string& name) const noexcept;
    [[nodiscard]] const GgufMetadataValue* find_metadata(const std::string& key) const noexcept;
    [[nodiscard]] const std::unordered_map<std::string, GgufTensorInfo>& tensor_infos() const noexcept;
    [[nodiscard]] const std::unordered_map<std::string, GgufMetadataValue>& metadata_values() const noexcept;
    [[nodiscard]] const std::filesystem::path& source_path() const noexcept;
    [[nodiscard]] const std::filesystem::path& root_path() const noexcept;

private:
    std::unordered_map<std::string, GgufTensorInfo> tensors;
    std::unordered_map<std::string, GgufMetadataValue> metadata;
    std::filesystem::path selected_source;
    std::filesystem::path selected_root;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GGUF_H
