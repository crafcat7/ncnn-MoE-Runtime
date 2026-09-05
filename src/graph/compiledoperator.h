#ifndef NCNN_MOE_COMPILEDOPERATOR_H
#define NCNN_MOE_COMPILEDOPERATOR_H

#include "storage/weightstore.h"

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
class NcnnVulkanQnkOperator;
class NcnnVulkanAttentionOperator;
class NcnnVulkanGatedDeltaNetOperator;
struct Mxfp4Q8PackedMatrix;
struct QnKPack;

using CompiledOperatorHandle = uint32_t;
inline constexpr CompiledOperatorHandle invalid_compiled_operator_handle = std::numeric_limits<CompiledOperatorHandle>::max();

// Execution products are owned by the resident model or an Expert-cache
// Entry; TensorData contains only weight storage.
struct CompiledOperator
{
    std::shared_ptr<NcnnLinearOperator> linear;
    std::shared_ptr<NcnnVulkanBfloat16Operator> bfloat16;
    std::shared_ptr<NcnnVulkanFloat8Operator> float8;
    std::shared_ptr<NcnnVulkanQnkOperator> qnk;
    std::shared_ptr<NcnnVulkanAttentionOperator> attention;
    std::shared_ptr<NcnnVulkanGatedDeltaNetOperator> gated_delta;

    // Created lazily when the explicit CPU packed-weight option is enabled.
    mutable std::shared_ptr<const Mxfp4Q8PackedMatrix> mxfp4_q8_packed;
    mutable std::shared_ptr<const QnKPack> qnk_packed;
};

class CompiledOperatorTable
{
public:
    void bind_weight_count(size_t count)
    {
        entries.clear();
        weight_count = 0;
        for (; weight_count < count; ++weight_count)
            (void)allocate();
    }

    [[nodiscard]] CompiledOperatorHandle allocate()
    {
        const CompiledOperatorHandle handle = static_cast<CompiledOperatorHandle>(entries.size());
        entries.emplace_back();
        return handle;
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return entries.size();
    }

    [[nodiscard]] CompiledOperatorHandle handle_for_weight(TensorHandle handle) const noexcept
    {
        return handle < weight_count
                   ? handle
                   : invalid_compiled_operator_handle;
    }

    [[nodiscard]] const CompiledOperator& at(CompiledOperatorHandle handle) const noexcept
    {
        return handle < entries.size() ? entries[handle] : empty_operator;
    }

    [[nodiscard]] CompiledOperator& at_mutable(CompiledOperatorHandle handle)
    {
        assert(handle != invalid_compiled_operator_handle);
        assert(handle < entries.size());
        return entries[handle];
    }

    [[nodiscard]] const CompiledOperator& at_weight(TensorHandle handle) const noexcept
    {
        return at(handle_for_weight(handle));
    }

    [[nodiscard]] const CompiledOperator* find_weight(TensorHandle handle) const noexcept
    {
        // CPU lazy caches must not use the shared empty_operator as an owner.
        return handle < weight_count ? &entries[handle] : nullptr;
    }

    [[nodiscard]] CompiledOperator& at_weight_mutable(TensorHandle handle)
    {
        const CompiledOperatorHandle compiled_handle = handle_for_weight(handle);
        assert(compiled_handle != invalid_compiled_operator_handle);
        return at_mutable(compiled_handle);
    }

private:
    std::vector<CompiledOperator> entries;
    size_t weight_count = 0;
    CompiledOperator empty_operator;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_COMPILEDOPERATOR_H
