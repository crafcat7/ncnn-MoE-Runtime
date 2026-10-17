#ifndef NCNN_MOE_GGUFMODEL_GPTOSS_H
#define NCNN_MOE_GGUFMODEL_GPTOSS_H

#include "ggufmodel.h"

namespace ncnn {
namespace moe {

[[nodiscard]] Result<std::string> gguf_gpt_oss_manifest(const GgufArchive& archive);
[[nodiscard]] Result<std::unordered_map<std::string, SafetensorInfo>> gguf_gpt_oss_tensors(const GgufArchive& archive);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GGUFMODEL_GPTOSS_H
