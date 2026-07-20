#include "ncnn/moe/session.h"

#include "cpu_executor.h"
#include "cpu_session_state.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace ncnn {
namespace moe {

Session::Session(ModelPtr model)
    : model_(std::move(model)), state_(new CpuSessionState)
{
    statistics_.expert_token_counts.resize(model_->descriptor().expert_count, 0);
}

Session::~Session() = default;

Result<PrefillResult> Session::prefill(std::span<const int32_t> input_ids)
{
    if (input_ids.empty())
        return Error{ErrorCode::InvalidArgument, "prefill requires at least one token"};
    if (input_ids.size() > std::numeric_limits<uint32_t>::max())
        return Error{ErrorCode::InvalidArgument, "prefill token count exceeds uint32 range"};
    const uint32_t max_context_length = model_->descriptor().layers.empty()
                                            ? 0
                                            : model_->descriptor().layers[0].attention.max_context_length;
    if (max_context_length > 0 && input_ids.size() > max_context_length - std::min<uint64_t>(sequence_length_, max_context_length))
        return Error{ErrorCode::InvalidArgument, "prefill exceeds the model context length"};

    SessionStatistics updated_statistics = statistics_;
    CpuExecutor executor;
    auto all_logits = executor.execute(*model_->compiled_, input_ids, updated_statistics, *state_, sequence_length_);
    if (!all_logits)
        return all_logits.error();

    statistics_ = std::move(updated_statistics);
    statistics_.prefill_tokens += input_ids.size();
    sequence_length_ += input_ids.size();

    PrefillResult result;
    result.logits.values = std::move(all_logits.value().back());
    result.processed_tokens = static_cast<uint32_t>(input_ids.size());
    return result;
}

Result<DecodeResult> Session::decode(int32_t input_id)
{
    const uint32_t max_context_length = model_->descriptor().layers.empty()
                                            ? 0
                                            : model_->descriptor().layers[0].attention.max_context_length;
    if (max_context_length > 0 && sequence_length_ >= max_context_length)
        return Error{ErrorCode::InvalidArgument, "decode exceeds the model context length"};
    SessionStatistics updated_statistics = statistics_;
    const std::span<const int32_t> input(&input_id, 1);
    CpuExecutor executor;
    auto all_logits = executor.execute(*model_->compiled_, input, updated_statistics, *state_, sequence_length_);
    if (!all_logits)
        return all_logits.error();

    statistics_ = std::move(updated_statistics);
    ++statistics_.decode_tokens;
    ++sequence_length_;

    DecodeResult result;
    result.logits.values = std::move(all_logits.value().front());
    result.sequence_length = sequence_length_;
    return result;
}

Result<void> Session::reset()
{
    sequence_length_ = 0;
    statistics_ = {};
    statistics_.expert_token_counts.resize(model_->descriptor().expert_count, 0);
    state_.reset(new CpuSessionState);
    return {};
}

} // namespace moe
} // namespace ncnn
