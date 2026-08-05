#include "ncnn/moe/weight_store.h"

#include <cassert>
#include <utility>

namespace ncnn {
namespace moe {

Result<TensorHandle> WeightStore::add(std::string name, TensorData tensor)
{
    if (name.empty())
        return Error{ErrorCode::InvalidArgument, "tensor name cannot be empty"};
    if (handles_.contains(name))
        return Error{ErrorCode::InvalidModel, "duplicate tensor: " + name};

    const TensorHandle handle = static_cast<TensorHandle>(tensors_.size());
    tensors_.push_back(std::move(tensor));
    handles_.emplace(std::move(name), handle);
    return handle;
}

const TensorData& WeightStore::at(TensorHandle handle) const
{
    assert(handle < tensors_.size());
    return tensors_[handle];
}

TensorData& WeightStore::at_mutable(TensorHandle handle)
{
    assert(handle < tensors_.size());
    return tensors_[handle];
}

TensorHandle WeightStore::find_handle(const std::string& name) const noexcept
{
    const auto it = handles_.find(name);
    return it == handles_.end() ? invalid_tensor_handle : it->second;
}

} // namespace moe
} // namespace ncnn
