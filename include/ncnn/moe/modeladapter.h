#ifndef NCNN_MOE_MODELADAPTER_H
#define NCNN_MOE_MODELADAPTER_H

#include "ncnn/moe/modeldescriptor.h"
#include "ncnn/moe/result.h"

#include <filesystem>
#include <string>

namespace ncnn {
namespace moe {

struct ModelManifest
{
    std::string model_type;
    std::string raw_json;
};

#define NCNN_MOE_MODEL_DEFER_MXFP4_BIT 0

enum ModelPackageFlag : uint32_t
{
    ModelPackageDeferMxfp4Experts = UINT32_C(1) << NCNN_MOE_MODEL_DEFER_MXFP4_BIT
};

struct ModelPackage
{
    std::filesystem::path root;
    ModelManifest manifest;
    uint32_t flags = 0;
};

class ModelAdapter
{
public:
    virtual ~ModelAdapter() = default;

    [[nodiscard]] virtual bool can_load(const ModelManifest& manifest) const = 0;

    [[nodiscard]] virtual Result<MoeModelDescriptor> parse_model(const ModelPackage& package) const = 0;

    [[nodiscard]] virtual Result<WeightMapping> map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELADAPTER_H
