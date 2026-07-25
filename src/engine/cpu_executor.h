#ifndef NCNN_MOE_CPU_EXECUTOR_H
#define NCNN_MOE_CPU_EXECUTOR_H

#include "ncnn/moe/execution_plan.h"
#include "ncnn/moe/result.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct SessionStatistics;
class CpuSessionState;

class IExecutor
{
public:
    virtual ~IExecutor() = default;

    [[nodiscard]] virtual Result<std::vector<std::vector<float> > > execute(
        const CompiledModel& model,
        std::span<const int32_t> input_ids,
        SessionStatistics& statistics,
        CpuSessionState& state,
        uint64_t position_offset) const
        = 0;
};

class CpuExecutor final : public IExecutor
{
public:
    [[nodiscard]] Result<std::vector<std::vector<float> > > execute(
        const CompiledModel& model,
        std::span<const int32_t> input_ids,
        SessionStatistics& statistics,
        CpuSessionState& state,
        uint64_t position_offset) const override;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_EXECUTOR_H
