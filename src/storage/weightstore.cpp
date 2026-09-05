#include "weightstore.h"

#include <utility>

namespace ncnn {
namespace moe {

Result<TensorHandle> WeightStore::add(std::string name, TensorData tensor)
{
    if (name.empty())
        return Error{ErrorCode::InvalidArgument, "tensor name cannot be empty"};
    if (handles.contains(name))
        return Error{ErrorCode::InvalidModel, "duplicate tensor: " + name};

    const TensorHandle handle = static_cast<TensorHandle>(tensors.size());
    tensors.push_back(std::move(tensor));
    handles.emplace(std::move(name), handle);
    return handle;
}

TensorHandle WeightStore::find_handle(const std::string& name) const noexcept
{
    const auto it = handles.find(name);
    return it == handles.end() ? invalid_tensor_handle : it->second;
}

} // namespace moe
} // namespace ncnn
