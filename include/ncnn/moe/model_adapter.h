#ifndef NCNN_MOE_MODEL_ADAPTER_H
#define NCNN_MOE_MODEL_ADAPTER_H

#include "ncnn/moe/moe_ir.h"
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

enum ModelPackageFlag : uint32_t
{
    ModelPackageDeferMxfp4Experts = 1u << 0
};

struct ModelPackage
{
    std::filesystem::path root;
    ModelManifest manifest;
    uint32_t flags = 0;
};

class IMoeModelAdapter
{
public:
    virtual ~IMoeModelAdapter() = default;

    [[nodiscard]] virtual bool can_load(const ModelManifest& manifest) const = 0;

    [[nodiscard]] virtual Result<MoeIR> parse_model(
        const ModelPackage& package) const
        = 0;

    [[nodiscard]] virtual Result<WeightMapping> map_weights(
        const ModelPackage& package,
        const MoeIR& ir) const
        = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODEL_ADAPTER_H
