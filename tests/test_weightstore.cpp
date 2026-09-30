#include "storage/weightstore.h"

#include <stdexcept>
#include <utility>

namespace ncnn {
namespace moe {

static void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void test_weightstore_finalization()
{
    WeightStore weights;
    TensorData first;
    first.shape = {1};
    first.float32_data = {1.25f};
    auto first_result = weights.add("first", std::move(first));
    require(static_cast<bool>(first_result), "first tensor was not added");
    const TensorHandle first_handle = first_result.value();

    TensorData second;
    second.shape = {2};
    second.float32_data = {2.5f, 3.5f};
    auto second_result = weights.add("second", std::move(second));
    require(static_cast<bool>(second_result), "second tensor was not added");
    const TensorHandle second_handle = second_result.value();
    const TensorData* first_data = &weights.at(first_handle);
    const TensorData* second_data = &weights.at(second_handle);

    auto duplicate = weights.add("first", TensorData{});
    require(!duplicate, "duplicate tensor was accepted");
    require(duplicate.error().code == ErrorCode::InvalidModel, "duplicate tensor returned the wrong error");
    require(weights.find_handle("first") == first_handle, "first tensor handle changed before finalization");
    require(weights.find_handle("second") == second_handle, "second tensor handle changed before finalization");

    weights.finalize();
    require(weights.is_finalized(), "weight store did not finalize");
    require(weights.size() == 2, "finalization changed tensor count");
    require(weights.find_handle("first") == invalid_tensor_handle, "released index still resolved first tensor");
    require(weights.find_handle("second") == invalid_tensor_handle, "released index still resolved second tensor");
    require(&weights.at(first_handle) == first_data, "first tensor address changed during finalization");
    require(&weights.at(second_handle) == second_data, "second tensor address changed during finalization");
    require(weights.at(first_handle).float32_values()[0] == 1.25f, "first tensor data changed during finalization");
    require(weights.at(second_handle).float32_values()[1] == 3.5f, "second tensor data changed during finalization");

    weights.finalize();
    require(weights.is_finalized(), "idempotent finalization cleared the finalized state");
    auto after_finalize = weights.add("third", TensorData{});
    require(!after_finalize, "tensor was accepted after finalization");
    require(after_finalize.error().code == ErrorCode::InvalidArgument, "post-finalization add returned the wrong error");
    require(weights.size() == 2, "post-finalization add changed tensor count");
}

} // namespace moe
} // namespace ncnn
