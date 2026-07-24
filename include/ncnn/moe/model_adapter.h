#ifndef NCNN_MOE_MODEL_ADAPTER_H
#define NCNN_MOE_MODEL_ADAPTER_H

#include "ncnn/moe/model_descriptor.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <filesystem>
#include <string>

namespace ncnn {
namespace moe {

struct ModelManifest
{
    std::string model_type;
    std::string raw_json;
};

struct ModelPackage
{
    std::filesystem::path root;
    ModelManifest manifest;
    bool defer_mxfp4_experts = false;
};

class IMoeModelAdapter
{
public:
    virtual ~IMoeModelAdapter() = default;

    [[nodiscard]] virtual bool can_load(const ModelManifest& manifest) const = 0;

    [[nodiscard]] virtual Result<MoeModelDescriptor> parse_model(
        const ModelPackage& package) const
        = 0;

    [[nodiscard]] virtual Result<WeightMapping> map_weights(
        const ModelPackage& package,
        const MoeModelDescriptor& descriptor) const
        = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODEL_ADAPTER_H
