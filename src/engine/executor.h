#ifndef NCNN_MOE_EXECUTOR_H
#define NCNN_MOE_EXECUTOR_H

#include "ncnn/moe/result.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct SessionStatistics;
struct CompiledModel;
class CpuSessionState;

struct CpuDecodeBatchEntry
{
    int32_t input_id = -1;
    SessionStatistics* statistics = nullptr;
    CpuSessionState* state = nullptr;
    uint64_t position_offset = 0;
};

struct CpuSpeculativeProposal
{
    std::vector<int32_t> token_ids;
    std::vector<std::vector<float>> logits;
    std::vector<float> confidence_logits;
    size_t committed_context_rows = 0;
};

using CpuSpeculativeSampler = std::function<Result<int32_t>(const std::vector<float>& logits)>;

[[nodiscard]] Result<std::vector<std::vector<float>>> forward_model(const CompiledModel& model, std::span<const int32_t> input_ids, SessionStatistics& statistics, CpuSessionState& state, uint64_t position_offset);

[[nodiscard]] Result<std::vector<std::vector<float>>> forward_decode_batch(const CompiledModel& model, std::span<const CpuDecodeBatchEntry> entries);

[[nodiscard]] Result<void> update_speculative_context(
    const CompiledModel& model,
    SessionStatistics& statistics,
    CpuSessionState& state);

[[nodiscard]] Result<CpuSpeculativeProposal> propose_speculative(
    const CompiledModel& model,
    int32_t input_id,
    SessionStatistics& statistics,
    CpuSessionState& state,
    uint64_t position_offset,
    const CpuSpeculativeSampler& sampler);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXECUTOR_H
