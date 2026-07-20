#ifndef NCNN_MOE_SESSION_H
#define NCNN_MOE_SESSION_H

#include "ncnn/moe/model.h"
#include "ncnn/moe/result.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

class CpuSessionState;

struct LogitsOutput
{
    std::vector<float> values;
};

struct PrefillResult
{
    LogitsOutput logits;
    uint32_t processed_tokens = 0;
};

struct DecodeResult
{
    LogitsOutput logits;
    uint64_t sequence_length = 0;
};

struct SessionStatistics
{
    uint64_t prefill_tokens = 0;
    uint64_t decode_tokens = 0;
    uint64_t expert_assignments = 0;
    uint64_t expert_batches = 0;
    std::vector<uint64_t> expert_token_counts;
};

struct SessionOptions
{
    LogitsOutputMode logits_output_mode = LogitsOutputMode::FullLogits;
};

class Session
{
public:
    ~Session();

    [[nodiscard]] Result<PrefillResult> prefill(std::span<const int32_t> input_ids);
    [[nodiscard]] Result<DecodeResult> decode(int32_t input_id);
    [[nodiscard]] Result<void> reset();

    [[nodiscard]] uint64_t sequence_length() const noexcept
    {
        return sequence_length_;
    }
    [[nodiscard]] const SessionStatistics& statistics() const noexcept
    {
        return statistics_;
    }

private:
    explicit Session(ModelPtr model);

    ModelPtr model_;
    uint64_t sequence_length_ = 0;
    SessionStatistics statistics_;
    std::unique_ptr<CpuSessionState> state_;

    friend class Runtime;
};

using SessionPtr = std::shared_ptr<Session>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SESSION_H
