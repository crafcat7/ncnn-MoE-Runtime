#ifndef NCNN_MOE_DEEPSEEK_V4_MODEL_ADAPTER_H
#define NCNN_MOE_DEEPSEEK_V4_MODEL_ADAPTER_H

#include "ncnn/moe/model_adapter.h"

namespace ncnn {
namespace moe {

class DeepSeekV4ModelAdapter final : public IMoeModelAdapter
{
public:
    [[nodiscard]] bool can_load(const ModelManifest& manifest) const override;
    [[nodiscard]] Result<MoeIR> parse_model(const ModelPackage& package) const override;
    [[nodiscard]] Result<WeightMapping> map_weights(const ModelPackage& package, const MoeIR& ir) const override;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_DEEPSEEK_V4_MODEL_ADAPTER_H
