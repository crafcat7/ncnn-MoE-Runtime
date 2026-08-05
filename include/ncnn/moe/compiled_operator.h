#ifndef NCNN_MOE_COMPILED_OPERATOR_H
#define NCNN_MOE_COMPILED_OPERATOR_H

#include "ncnn/moe/tensor_handle.h"

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <limits>
#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

class NcnnLinearOperator;
class NcnnVulkanBfloat16Operator;
class NcnnVulkanFloat8Operator;
class NcnnVulkanAttentionOperator;
class NcnnVulkanGatedDeltaNetOperator;

using CompiledOperatorHandle = uint32_t;
inline constexpr CompiledOperatorHandle invalid_compiled_operator_handle = std::numeric_limits<CompiledOperatorHandle>::max();

// Backend compilation products are owned by the compiled model, never by the
// immutable weight storage.  Keeping this table separate also lets CPU-only
// models retain no Vulkan object through TensorData.
struct CompiledOperator
{
    std::shared_ptr<NcnnLinearOperator> linear;
    std::shared_ptr<NcnnVulkanBfloat16Operator> bfloat16;
    std::shared_ptr<NcnnVulkanFloat8Operator> float8;
    std::shared_ptr<NcnnVulkanAttentionOperator> attention;
    std::shared_ptr<NcnnVulkanGatedDeltaNetOperator> gated_delta;
};

class CompiledOperatorTable
{
private:
    std::vector<CompiledOperator> entries_;
    std::vector<CompiledOperatorHandle> weight_bindings_;

public:
    void bind_weight_count(size_t count)
    {
        entries_.clear();
        weight_bindings_.clear();
        weight_bindings_.reserve(count);
        for (size_t index = 0; index < count; ++index)
            weight_bindings_.push_back(allocate());
    }

    [[nodiscard]] CompiledOperatorHandle allocate()
    {
        const CompiledOperatorHandle handle = static_cast<CompiledOperatorHandle>(entries_.size());
        entries_.emplace_back();
        return handle;
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return entries_.size();
    }

    [[nodiscard]] CompiledOperatorHandle handle_for_weight(TensorHandle handle) const noexcept
    {
        return handle < weight_bindings_.size()
                   ? weight_bindings_[handle]
                   : invalid_compiled_operator_handle;
    }

    [[nodiscard]] const CompiledOperator& at(CompiledOperatorHandle handle) const noexcept
    {
        return handle < entries_.size() ? entries_[handle] : empty_;
    }

    [[nodiscard]] CompiledOperator& at_mutable(CompiledOperatorHandle handle)
    {
        assert(handle != invalid_compiled_operator_handle);
        assert(handle < entries_.size());
        return entries_[handle];
    }

    [[nodiscard]] const CompiledOperator& at_weight(TensorHandle handle) const noexcept
    {
        return at(handle_for_weight(handle));
    }

    [[nodiscard]] CompiledOperator& at_weight_mutable(TensorHandle handle)
    {
        const CompiledOperatorHandle compiled_handle = handle_for_weight(handle);
        assert(compiled_handle != invalid_compiled_operator_handle);
        return at_mutable(compiled_handle);
    }

private:
    CompiledOperator empty_;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_COMPILED_OPERATOR_H
