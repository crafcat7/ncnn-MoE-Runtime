#ifndef NCNN_MOE_GGUFMODEL_QWEN_H
#define NCNN_MOE_GGUFMODEL_QWEN_H

#include "gguf.h"
#include "safetensors.h"

namespace ncnn {
namespace moe {

[[nodiscard]] Result<std::string> gguf_qwen_manifest(const GgufArchive& archive);
[[nodiscard]] Result<std::unordered_map<std::string, SafetensorInfo>> gguf_qwen_tensors(const GgufArchive& archive);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GGUFMODEL_QWEN_H
