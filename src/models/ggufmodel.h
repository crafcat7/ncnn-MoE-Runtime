#ifndef NCNN_MOE_GGUFMODEL_H
#define NCNN_MOE_GGUFMODEL_H

#include "gguf.h"
#include "safetensors.h"

#include <string_view>

namespace ncnn {
namespace moe {

[[nodiscard]] bool has_gguf_source(const std::filesystem::path& path);
[[nodiscard]] Result<std::string> gguf_model_manifest(const GgufArchive& archive);
[[nodiscard]] Result<std::unordered_map<std::string, SafetensorInfo>> gguf_model_tensors(const GgufArchive& archive);
[[nodiscard]] bool gguf_layer_name(std::string_view name, std::string_view& number, std::string_view& suffix) noexcept;
[[nodiscard]] std::string gguf_storage_dtype(uint32_t type);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GGUFMODEL_H
