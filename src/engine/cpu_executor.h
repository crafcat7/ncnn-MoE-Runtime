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

struct CpuDecodeBatchEntry
{
    int32_t input_id = -1;
    SessionStatistics* statistics = nullptr;
    CpuSessionState* state = nullptr;
    uint64_t position_offset = 0;
};

struct CpuDecodeBatchMetrics
{
    uint64_t logical_expert_batches = 0;
    uint64_t physical_expert_batches = 0;
    uint64_t coalesced_expert_routes = 0;
    uint64_t max_expert_batch_size = 0;
};

class IExecutor
{
public:
    virtual ~IExecutor() = default;

    [[nodiscard]] virtual Result<std::vector<std::vector<float>>> execute(const CompiledModel& model, std::span<const int32_t> input_ids, SessionStatistics& statistics, CpuSessionState& state, uint64_t position_offset) const = 0;
};

class CpuExecutor final : public IExecutor
{
public:
    [[nodiscard]] Result<std::vector<std::vector<float>>> execute(const CompiledModel& model, std::span<const int32_t> input_ids, SessionStatistics& statistics, CpuSessionState& state, uint64_t position_offset) const override;

    [[nodiscard]] Result<std::vector<std::vector<float>>> execute_decode_batch(const CompiledModel& model, std::span<const CpuDecodeBatchEntry> entries, CpuDecodeBatchMetrics& metrics) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_EXECUTOR_H
