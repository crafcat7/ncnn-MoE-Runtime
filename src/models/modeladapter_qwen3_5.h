#ifndef NCNN_MOE_MODELS_MODELADAPTER_QWEN3_5_H
#define NCNN_MOE_MODELS_MODELADAPTER_QWEN3_5_H

#include "ncnn/moe/modeladapter.h"

namespace ncnn {
namespace moe {

class Qwen3_5MoeModelAdapter final : public ModelAdapter
{
public:
    [[nodiscard]] bool can_load(const ModelManifest& manifest) const override;
    [[nodiscard]] Result<MoeModelDescriptor> parse_model(const ModelPackage& package) const override;
    [[nodiscard]] Result<WeightMapping> map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const override;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELS_MODELADAPTER_QWEN3_5_H
