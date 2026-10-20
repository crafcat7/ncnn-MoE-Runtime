#ifndef NCNN_MOE_EXECUTOR_H
#define NCNN_MOE_EXECUTOR_H

#include "kernels/activationbuffer.h"
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
struct BatchWorkspace;
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
    // Distinct caller-owned outputs must remain valid during execution; null skips LM output.
    std::vector<float>* logits = nullptr;
};

struct SpeculativeProposal
{
    std::vector<int32_t> token_ids;
    std::vector<ActivationBuffer> logits;
    std::vector<float> confidence_logits;
    size_t committed_context_rows = 0;
};

using SpeculativeSampler = std::function<Result<int32_t>(std::span<const float> logits)>;

// Execute the model into the session's reusable logits buffer.  Callers must
// consume state.logits before another execution on the same session.
[[nodiscard]] Result<void> forward_model(const CompiledModel& model,
                                         std::span<const int32_t> input_ids,
                                         SessionStatistics& statistics,
                                         SessionState& state,
                                         uint64_t position_offset,
                                         LogitsOutput logits_output = LogitsOutput::All);

[[nodiscard]] Result<void> forward_decode_batch(const CompiledModel& model,
                                                std::span<const DecodeBatchEntry> entries,
                                                BatchWorkspace& workspace);

[[nodiscard]] Result<void> update_speculative_context(const CompiledModel& model,
                                                      SessionStatistics& statistics,
                                                      SessionState& state);

// Reuses logits storage; the proposal is consumed only after a successful result.
[[nodiscard]] Result<void> propose_speculative(const CompiledModel& model,
                                               int32_t input_id,
                                               SessionStatistics& statistics,
                                               SessionState& state,
                                               uint64_t position_offset,
                                               SpeculativeProposal& proposal,
                                               const SpeculativeSampler& sampler);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXECUTOR_H
