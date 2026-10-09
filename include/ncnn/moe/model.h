#ifndef NCNN_MOE_MODEL_H
#define NCNN_MOE_MODEL_H

#include "ncnn/moe/modeldescriptor.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ncnn {
namespace moe {

struct CompiledModel;

class Model
{
public:
    [[nodiscard]] const MoeModelDescriptor& descriptor() const noexcept;
    [[nodiscard]] HybridMode hybrid_mode() const noexcept;
    [[nodiscard]] uint32_t vulkan_device_index() const noexcept;
    [[nodiscard]] const std::vector<uint32_t>& vulkan_device_indices() const noexcept;

    // Format a UTF-8 JSON array of text messages with the model's chat template.
    // Invalid UTF-8 is rejected; callers handle console or locale conversion.
    // Tokenizer assets are loaded once with the model. An empty stop list
    // means native text handling is unavailable; token-ID execution still works.
    [[nodiscard]] Result<std::vector<int32_t>> encode(std::string_view messages, bool enable_thinking = true) const;
    // Each text stream owns its pending UTF-8 bytes. Pass -1 and final=true
    // to flush an incomplete final character without appending another token.
    [[nodiscard]] Result<std::string> decode(int32_t token_id, std::string& pending, bool final = false) const;
    [[nodiscard]] const std::vector<int32_t>& stop_tokens() const noexcept;

private:
    explicit Model(std::shared_ptr<const CompiledModel> _compiled);

    std::shared_ptr<const CompiledModel> compiled;

    friend class Runtime;
    friend const CompiledModel& model_compiled(const Model& model) noexcept;
};

using ModelPtr = std::shared_ptr<Model>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODEL_H
