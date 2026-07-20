#ifndef NCNN_MOE_NCNN_LINEAR_H
#define NCNN_MOE_NCNN_LINEAR_H

#include "cpu_batch.h"

#include "ncnn/moe/types.h"

#include <memory>

namespace ncnn {
namespace moe {

class NcnnLinearOperator
{
public:
    ~NcnnLinearOperator();

    [[nodiscard]] static std::shared_ptr<NcnnLinearOperator> create(
        const TensorData& matrix,
        const TensorData* bias);
    [[nodiscard]] bool forward(const CpuBatch& input, CpuBatch& output) const;

private:
    class Implementation;

    NcnnLinearOperator();
    std::unique_ptr<Implementation> implementation_;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NCNN_LINEAR_H
