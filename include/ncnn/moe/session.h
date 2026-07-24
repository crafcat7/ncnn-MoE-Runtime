#ifndef NCNN_MOE_SESSION_H
#define NCNN_MOE_SESSION_H

#include "ncnn/moe/model.h"
#include "ncnn/moe/result.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

class CpuSessionState;
class IExecutor;

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

struct SamplingOptions
{
    // A zero temperature selects the highest-logit token deterministically.
    float temperature = 1.0f;
    // Zero keeps every vocabulary item. Non-zero keeps the highest-logit items.
    uint32_t top_k = 0;
    // Nucleus sampling threshold in the range (0, 1].
    float top_p = 1.0f;
    // Removes tokens below this fraction of the most likely token's probability.
    float min_p = 0.0f;
};

struct SampledToken
{
    int32_t token_id = -1;
    float probability = 0.0f;
};

struct StreamToken
{
    uint32_t index = 0;
    int32_t token_id = -1;
    float probability = 0.0f;
    std::string text;
    bool is_stop_token = false;
};

struct GenerationOptions
{
    uint32_t max_new_tokens = 1;
    SamplingOptions sampling;
    std::vector<int32_t> stop_tokens;
};

struct GenerationResult
{
    std::vector<StreamToken> tokens;
    bool stopped_by_stop_token = false;
    bool stopped_by_callback = false;
};

using TokenTextDecoder = std::function<std::string(int32_t token_id)>;
using TokenStreamCallback = std::function<bool(const StreamToken& token)>;

struct SessionStatistics
{
    uint64_t prefill_tokens = 0;
    uint64_t decode_tokens = 0;
    uint64_t expert_assignments = 0;
    uint64_t expert_batches = 0;
    uint64_t expert_prefetches = 0;
    uint64_t expert_prefetch_bytes = 0;
    uint64_t expert_parallel_tasks = 0;
    uint64_t mxfp4_decode_gemv_rows = 0;
    uint64_t mxfp4_prefill_gemm_rows = 0;
    uint64_t mxfp4_paired_rows = 0;
    uint64_t mxfp4_fused_gate_up_rows = 0;
    uint64_t vulkan_linear_dispatches = 0;
    uint64_t vulkan_attention_blocks = 0;
    uint64_t vulkan_compute_submissions = 0;
    uint64_t vulkan_batch_uploads = 0;
    uint64_t vulkan_batch_downloads = 0;
    uint64_t vulkan_auxiliary_uploads = 0;
    uint64_t vulkan_auxiliary_upload_bytes = 0;
    uint64_t vulkan_staging_slot_resizes = 0;
    uint64_t vulkan_staging_slot_reuses = 0;
    uint64_t vulkan_staging_slot_acquisitions = 0;
    uint64_t vulkan_staging_slot_contentions = 0;
    uint64_t attention_time_microseconds = 0;
    uint64_t router_time_microseconds = 0;
    uint64_t expert_time_microseconds = 0;
    uint64_t kv_cache_logical_bytes = 0;
    uint64_t kv_cache_allocated_bytes = 0;
    std::vector<uint64_t> expert_token_counts;
};

struct SessionOptions
{
    LogitsOutputMode logits_output_mode = LogitsOutputMode::FullLogits;
    // A deterministic default makes generation reproducible unless callers opt in to another seed.
    uint64_t sampling_seed = 0;
    // Bounds Prefill working memory. Zero processes the whole prompt in one batch.
    uint32_t prefill_chunk_size = 256;
};

class Session
{
public:
    ~Session();

    [[nodiscard]] Result<PrefillResult> prefill(std::span<const int32_t> input_ids);
    [[nodiscard]] Result<DecodeResult> decode(int32_t input_id);
    [[nodiscard]] Result<SampledToken> sample(
        const LogitsOutput& logits,
        const SamplingOptions& options = {});
    [[nodiscard]] Result<GenerationResult> generate(
        std::span<const int32_t> input_ids,
        const GenerationOptions& options = {},
        TokenStreamCallback on_token = {},
        TokenTextDecoder decode_text = {});
    [[nodiscard]] Result<void> reset();

    [[nodiscard]] uint64_t sequence_length() const noexcept
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return sequence_length_;
    }
    [[nodiscard]] SessionStatistics statistics() const
    {
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        return statistics_;
    }

private:
    explicit Session(ModelPtr model, const SessionOptions& options);

    ModelPtr model_;
    uint64_t sequence_length_ = 0;
    SessionStatistics statistics_;
    std::unique_ptr<CpuSessionState> state_;
    std::unique_ptr<IExecutor> executor_;
    std::mt19937_64 random_generator_;
    uint32_t prefill_chunk_size_ = 256;
    mutable std::recursive_mutex mutex_;

    friend class Runtime;
};

using SessionPtr = std::shared_ptr<Session>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SESSION_H
