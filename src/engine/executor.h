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
class SessionState;

// Select LM Head outputs without skipping state updates for any input token.
enum class LogitsOutput
{
    None,
    Last,
    All
};

struct DecodeBatchEntry
{
    int32_t input_id = -1;
    SessionStatistics* statistics = nullptr;
    SessionState* state = nullptr;
    uint64_t position_offset = 0;
    bool output_logits = true;
};

struct SpeculativeProposal
{
    std::vector<int32_t> token_ids;
    std::vector<std::vector<float>> logits;
    std::vector<float> confidence_logits;
    size_t committed_context_rows = 0;
};

using SpeculativeSampler = std::function<Result<int32_t>(const std::vector<float>& logits)>;

[[nodiscard]] Result<std::vector<std::vector<float>>> forward_model(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    SessionStatistics& statistics,
    SessionState& state,
    uint64_t position_offset,
    LogitsOutput logits_output = LogitsOutput::All);

[[nodiscard]] Result<std::vector<std::vector<float>>> forward_decode_batch(const CompiledModel& model, std::span<const DecodeBatchEntry> entries);

[[nodiscard]] Result<void> update_speculative_context(
    const CompiledModel& model,
    SessionStatistics& statistics,
    SessionState& state);

[[nodiscard]] Result<SpeculativeProposal> propose_speculative(
    const CompiledModel& model,
    int32_t input_id,
    SessionStatistics& statistics,
    SessionState& state,
    uint64_t position_offset,
    const SpeculativeSampler& sampler);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXECUTOR_H
