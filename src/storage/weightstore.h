#ifndef NCNN_MOE_WEIGHTSTORE_H
#define NCNN_MOE_WEIGHTSTORE_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

using TensorHandle = uint32_t;
inline constexpr TensorHandle invalid_tensor_handle = std::numeric_limits<TensorHandle>::max();

// Immutable model weights are owned by this store.  Backend compilation
// products are deliberately kept in CompiledOperatorTable and never attached
// to TensorData entries.
class WeightStore
{
public:
    [[nodiscard]] Result<TensorHandle> add(std::string name, TensorData tensor);
    [[nodiscard]] const TensorData& at(TensorHandle handle) const
    {
        assert(handle < tensors.size());
        return tensors[handle];
    }
    [[nodiscard]] TensorData& at_mutable(TensorHandle handle)
    {
        assert(handle < tensors.size());
        return tensors[handle];
    }
    [[nodiscard]] TensorHandle find_handle(const std::string& name) const noexcept;
    [[nodiscard]] size_t size() const noexcept
    {
        return tensors.size();
    }

private:
    std::vector<TensorData> tensors;
    std::unordered_map<std::string, TensorHandle> handles;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_WEIGHTSTORE_H
