#ifndef NCNN_MOE_MODEL_H
#define NCNN_MOE_MODEL_H

#include "ncnn/moe/modeldescriptor.h"
#include "ncnn/moe/types.h"

#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

struct CompiledModel;

class Model
{
private:
    explicit Model(std::shared_ptr<const CompiledModel> _compiled);

    std::shared_ptr<const CompiledModel> compiled;

    friend class Runtime;
    friend const CompiledModel& model_compiled(const Model& model) noexcept;

public:
    [[nodiscard]] const MoeModelDescriptor& descriptor() const noexcept;
    [[nodiscard]] HybridMode hybrid_mode() const noexcept;
    [[nodiscard]] uint32_t vulkan_device_index() const noexcept;
    [[nodiscard]] const std::vector<uint32_t>& vulkan_device_indices() const noexcept;
};

using ModelPtr = std::shared_ptr<Model>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODEL_H
