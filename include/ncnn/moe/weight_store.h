#ifndef NCNN_MOE_WEIGHT_STORE_H
#define NCNN_MOE_WEIGHT_STORE_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/tensor_handle.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

// Immutable model weights are owned by this store.  Backend compilation
// products are deliberately kept in CompiledOperatorTable and never attached
// to TensorData entries.
class WeightStore
{
private:
    std::vector<TensorData> tensors_;
    std::unordered_map<std::string, TensorHandle> handles_;

public:
    [[nodiscard]] Result<TensorHandle> add(std::string name, TensorData tensor);
    [[nodiscard]] const TensorData& at(TensorHandle handle) const;
    [[nodiscard]] TensorData& at_mutable(TensorHandle handle);
    [[nodiscard]] TensorHandle find_handle(const std::string& name) const noexcept;
    [[nodiscard]] size_t size() const noexcept
    {
        return tensors_.size();
    }
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_WEIGHT_STORE_H
