#ifndef NCNN_MOE_SESSION_H
#define NCNN_MOE_SESSION_H

#include "ncnn/moe/model.h"
#include "ncnn/moe/result.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static constexpr uint32_t maximum_expert_route_ranks = 16;

class BatchSchedulerPrivate;

class SessionState;

struct PrefillResult
{
    std::vector<float> logits;
    uint32_t processed_tokens = 0;
};

struct DecodeResult
{
    std::vector<float> logits;
    uint64_t sequence_length = 0;
};

// Model-neutral token-ID distribution controls.
struct SamplingOptions
{
    float temperature = 1.0f;
    uint32_t top_k = 0;
    float top_p = 1.0f;
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
    // Optional decoded text supplied by the generation decoder.
    std::string text;
    bool is_stop_token = false;
};

// Token-ID generation controls; use Model::stop_tokens() for text generation.
struct GenerationOptions
{
    uint32_t max_new_tokens = 1;
    SamplingOptions sampling;
    std::vector<int32_t> stop_tokens;
    bool use_speculative = true;
    float speculative_confidence_threshold = 0.5f;
    uint32_t speculative_max_draft_tokens = 0;
};

struct GenerationResult
{
    std::vector<StreamToken> tokens;
    bool stopped_by_stop_token = false;
    bool stopped_by_callback = false;
};

// A generation decoder can use Model::decode with stream-local UTF-8 state.
using TokenTextDecoder = std::function<std::string(int32_t token_id)>;
using TokenStreamCallback = std::function<bool(const StreamToken& token)>;

struct SessionStatistics
{
    uint64_t prefill_tokens = 0;
    uint64_t decode_tokens = 0;
    uint64_t expert_assignments = 0;
    uint64_t expert_batches = 0;
    uint64_t expert_cpu_batches = 0;
    uint64_t expert_cpu_routes = 0;
    uint64_t expert_gpu_routes = 0;
    uint64_t expert_cpu_fallbacks = 0;
    uint64_t expert_batch_weight_bytes = 0;
    uint64_t expert_route_weight_bytes = 0;
    std::array<uint64_t, maximum_expert_route_ranks> expert_route_rank_predictions{};
    std::array<uint64_t, maximum_expert_route_ranks> expert_route_rank_matches{};
    std::array<uint64_t, maximum_expert_route_ranks> expert_route_rank_demands{};
    std::array<uint64_t, maximum_expert_route_ranks> expert_route_rank_demand_queue_time_microseconds{};
    uint64_t expert_cache_hits = 0;
    uint64_t expert_cache_misses = 0;
    uint64_t expert_cache_evictions = 0;
    uint64_t expert_cache_bytes_read = 0;
    uint64_t expert_cache_resident_size = 0;
    uint64_t expert_gpu_cache_hits = 0;
    uint64_t expert_gpu_cache_misses = 0;
    uint64_t expert_gpu_cache_stores = 0;
    uint64_t expert_gpu_cache_evictions = 0;
    uint64_t expert_gpu_cache_resident_size = 0;
    uint64_t expert_gpu_cache_pending_size = 0;
    uint64_t expert_gpu_executions = 0;
    uint64_t expert_gpu_execution_failures = 0;
    uint64_t vulkan_linear_dispatches = 0;
    uint64_t vulkan_int4_dispatches = 0;
    uint64_t vulkan_attention_blocks = 0;
    uint64_t vulkan_compute_submissions = 0;
    uint64_t vulkan_batch_uploads = 0;
    uint64_t vulkan_batch_downloads = 0;
    uint64_t vulkan_attention_qkv_rope_fusions = 0;
    uint64_t vulkan_attention_device_rope_fusions = 0;
    uint64_t vulkan_attention_qkv_ring_fusions = 0;
    uint64_t vulkan_attention_decode_sdpa_fusions = 0;
    uint64_t vulkan_attention_cpu_fallbacks = 0;
    uint64_t vulkan_kv_ring_appends = 0;
    uint64_t vulkan_kv_ring_resizes = 0;
    uint64_t vulkan_kv_ring_wrapped_views = 0;
    uint64_t vulkan_gated_delta_submissions = 0;
    uint64_t speculative_proposals = 0;
    uint64_t speculative_draft_tokens = 0;
    uint64_t speculative_accepted_tokens = 0;
    uint64_t kv_cache_logical_size = 0;
    uint64_t kv_cache_allocated_size = 0;
};

// Runtime counter snapshot used for generation and cumulative reporting.
struct RuntimeMetricCounters
{
    uint64_t prefill_tokens = 0;
    uint64_t decode_tokens = 0;
    uint64_t expert_assignments = 0;
    uint64_t expert_batches = 0;
    uint64_t expert_cpu_batches = 0;
    uint64_t expert_cpu_routes = 0;
    uint64_t expert_gpu_routes = 0;
    uint64_t expert_cpu_fallbacks = 0;
    uint64_t speculative_proposals = 0;
    uint64_t speculative_draft_tokens = 0;
    uint64_t speculative_accepted_tokens = 0;
    uint64_t expert_cache_hits = 0;
    uint64_t expert_cache_misses = 0;
    uint64_t expert_io_bytes = 0;
    uint64_t expert_cache_resident_size = 0;
    uint64_t expert_gpu_cache_hits = 0;
    uint64_t expert_gpu_cache_misses = 0;
    uint64_t expert_gpu_cache_stores = 0;
    uint64_t expert_gpu_cache_evictions = 0;
    uint64_t expert_gpu_cache_resident_size = 0;
    uint64_t expert_gpu_cache_pending_size = 0;
    uint64_t expert_gpu_executions = 0;
    uint64_t expert_gpu_execution_failures = 0;
    uint64_t gpu_submit_count = 0;
    uint64_t vulkan_linear_dispatches = 0;
    uint64_t vulkan_int4_dispatches = 0;
    uint64_t vulkan_attention_blocks = 0;
    uint64_t vulkan_batch_uploads = 0;
    uint64_t vulkan_batch_downloads = 0;
    uint64_t vulkan_attention_qkv_rope_fusions = 0;
    uint64_t vulkan_attention_device_rope_fusions = 0;
    uint64_t vulkan_attention_qkv_ring_fusions = 0;
    uint64_t vulkan_attention_cpu_fallbacks = 0;
    uint64_t vulkan_gated_delta_submissions = 0;
    uint64_t kv_cache_logical_size = 0;
    uint64_t kv_cache_allocated_size = 0;
};

struct GenerationTimingMetrics
{
    bool active = false;
    uint64_t input_tokens = 0;
    uint64_t output_tokens = 0;
    uint64_t elapsed_microseconds = 0;
    std::optional<uint64_t> prompt_elapsed_microseconds;
    std::optional<uint64_t> generation_elapsed_microseconds;
    std::optional<double> prompt_tokens_per_second;
    std::optional<double> generation_tokens_per_second;
    std::optional<uint64_t> ttft_microseconds;
    std::optional<double> tpot_microseconds;
    // Compatibility alias.
    std::optional<double> decode_tokens_per_second;
};

struct SessionMetrics
{
    RuntimeMetricCounters generation;
    RuntimeMetricCounters cumulative;
    GenerationTimingMetrics timing;
    bool gpu_available = false;
};

struct SessionOptions
{
    uint64_t sampling_seed = 0;
    uint32_t prefill_chunk_size = 512;
    bool use_speculative_context = true;
};

class Session
{
public:
    ~Session();

    // Argument validation leaves the session usable; an execution failure may
    // leave partial state and requires reset before another operation.
    [[nodiscard]] Result<PrefillResult> prefill(std::span<const int32_t> input_ids);
    [[nodiscard]] Result<DecodeResult> decode(int32_t input_id);
    [[nodiscard]] Result<SampledToken> sample(std::span<const float> logits, const SamplingOptions& opt = {});
    [[nodiscard]] Result<GenerationResult> generate(std::span<const int32_t> input_ids, const GenerationOptions& opt = {}, TokenStreamCallback on_token = {}, TokenTextDecoder decode_text = {});
    [[nodiscard]] Result<void> reset();

    [[nodiscard]] uint64_t sequence_length() const noexcept
    {
        const std::lock_guard<std::mutex> lock(mutex);
        return token_count;
    }
    [[nodiscard]] SessionStatistics statistics() const;
    [[nodiscard]] SessionMetrics metrics() const;

private:
    explicit Session(ModelPtr _model, const SessionOptions& opt);
    [[nodiscard]] static uint32_t context_limit(const MoeModelDescriptor& descriptor) noexcept;
    [[nodiscard]] Result<void> prefill_unlocked(std::span<const int32_t> input_ids);
    [[nodiscard]] Result<void> decode_unlocked(int32_t input_id);
    [[nodiscard]] Result<SampledToken> sample_unlocked(std::span<const float> logits, const SamplingOptions& opt);
    [[nodiscard]] Result<GenerationResult> generate_unlocked(std::span<const int32_t> input_ids, const GenerationOptions& opt, const TokenStreamCallback& on_token, const TokenTextDecoder& decode_text, std::unique_lock<std::mutex>& lock);
    [[nodiscard]] Result<GenerationResult> generate_speculative(const GenerationOptions& opt, const TokenStreamCallback& on_token, const TokenTextDecoder& decode_text, std::unique_lock<std::mutex>& lock);
    [[nodiscard]] Result<bool> emit_token(StreamToken& token, const TokenStreamCallback& on_token, const TokenTextDecoder& decode_text, std::unique_lock<std::mutex>& lock);
    void begin_generation(uint64_t input_tokens);
    void finish_generation() noexcept;
    // Resource counters are sampled on read, never on the execution path.
    [[nodiscard]] SessionStatistics snapshot_stats() const;
    // Publish counters and position; callers still own cache transactions.
    void commit_execution(uint64_t prefill_tokens, uint64_t decode_tokens);

    ModelPtr model;
    uint64_t token_count = 0;
    SessionStatistics stats;
    // Tentative execution counters are published when the step commits.
    SessionStatistics stats_scratch;
    std::unique_ptr<SessionState> state;
    std::mt19937_64 random_generator;
    uint32_t prefill_chunk_size = 512;
    bool use_speculative_context = true;
    RuntimeMetricCounters metrics_base;
    bool metrics_valid = false;
    bool generation_active = false;
    uint64_t generation_input_tokens = 0;
    uint64_t generation_output_tokens = 0;
    uint64_t generation_elapsed_microseconds = 0;
    std::chrono::steady_clock::time_point generation_start_time;
    std::chrono::steady_clock::time_point generation_first_token_time;
    mutable std::mutex mutex;
    std::vector<int32_t> sampling_token_ids;
    std::array<std::vector<SampledToken>, 2> sampling_candidates;
    std::vector<float> sampling_residual;

    friend class Runtime;
    friend class BatchSchedulerPrivate;
};

using SessionPtr = std::shared_ptr<Session>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SESSION_H
