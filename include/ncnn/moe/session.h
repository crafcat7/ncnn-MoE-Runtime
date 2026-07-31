#ifndef NCNN_MOE_SESSION_H
#define NCNN_MOE_SESSION_H

#include "ncnn/moe/memory_manager.h"
#include "ncnn/moe/model.h"
#include "ncnn/moe/result.h"

#include <array>
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

static constexpr uint32_t maximum_expert_route_ranks = 16;

class SessionBatchAccess;

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
    std::string text;
    bool is_stop_token = false;
};

struct GenerationOptions
{
    uint32_t max_new_tokens = 1;
    SamplingOptions sampling;
    std::vector<int32_t> stop_tokens;
    bool enable_speculative = true;
    float speculative_confidence_threshold = 0.5f;
    uint32_t speculative_max_draft_tokens = 0;
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
    uint64_t expert_batch_weight_bytes = 0;
    uint64_t expert_route_weight_bytes = 0;
    uint64_t expert_prefetches = 0;
    uint64_t expert_prefetch_bytes = 0;
    uint64_t expert_route_predictions = 0;
    uint64_t expert_route_prediction_matches = 0;
    uint64_t expert_route_prediction_cache_hits = 0;
    uint64_t expert_route_prediction_cache_misses = 0;
    uint64_t expert_route_prediction_time_microseconds = 0;
    uint64_t expert_route_prediction_wait_time_microseconds = 0;
    uint64_t expert_route_prediction_async_submissions = 0;
    uint64_t expert_route_prediction_async_completions = 0;
    uint64_t expert_route_prediction_async_fallbacks = 0;
    std::array<uint64_t, maximum_expert_route_ranks> expert_route_rank_predictions{};
    std::array<uint64_t, maximum_expert_route_ranks> expert_route_rank_matches{};
    std::array<uint64_t, maximum_expert_route_ranks> expert_route_rank_demands{};
    std::array<uint64_t, maximum_expert_route_ranks> expert_route_rank_demand_queue_time_microseconds{};
    uint64_t expert_cache_hits = 0;
    uint64_t expert_cache_misses = 0;
    uint64_t expert_cache_evictions = 0;
    uint64_t expert_cache_bytes_read = 0;
    uint64_t expert_cache_resident_bytes = 0;
    uint64_t expert_cache_queued_reads = 0;
    uint64_t expert_cache_speculative_reads = 0;
    uint64_t expert_cache_cancelled_speculative_reads = 0;
    uint64_t expert_cache_dropped_speculative_admissions = 0;
    uint64_t expert_cache_unused_speculative_reads = 0;
    uint64_t expert_cache_short_term_reloads = 0;
    uint64_t expert_cache_arc_recent_bytes = 0;
    uint64_t expert_cache_arc_frequent_bytes = 0;
    uint64_t expert_cache_arc_recent_target_bytes = 0;
    uint64_t expert_cache_arc_recent_ghost_bytes = 0;
    uint64_t expert_cache_arc_frequent_ghost_bytes = 0;
    uint64_t expert_cache_arc_recent_ghost_hits = 0;
    uint64_t expert_cache_arc_frequent_ghost_hits = 0;
    uint64_t expert_cache_mapped_ranges = 0;
    uint64_t expert_cache_mapped_bytes = 0;
    uint64_t expert_cache_direct_read_ranges = 0;
    uint64_t expert_cache_direct_read_bytes = 0;
    uint64_t expert_cache_direct_read_fallbacks = 0;
    uint64_t expert_cache_buffered_read_ranges = 0;
    uint64_t expert_cache_buffered_read_bytes = 0;
    uint64_t expert_cache_coalesced_read_batches = 0;
    uint64_t expert_cache_coalesced_experts = 0;
    uint64_t expert_cache_coalesced_read_ranges_saved = 0;
    uint32_t expert_cache_read_policy = 0;
    uint64_t expert_gpu_cache_hits = 0;
    uint64_t expert_gpu_cache_misses = 0;
    uint64_t expert_gpu_cache_admissions = 0;
    uint64_t expert_gpu_cache_stores = 0;
    uint64_t expert_gpu_cache_evictions = 0;
    uint64_t expert_gpu_cache_dropped_admissions = 0;
    uint64_t expert_gpu_cache_bytes_uploaded = 0;
    uint64_t expert_gpu_cache_resident_bytes = 0;
    uint64_t expert_gpu_cache_pending_bytes = 0;
    uint64_t expert_gpu_victim_cache_hits = 0;
    uint64_t expert_gpu_victim_cache_misses = 0;
    uint64_t expert_gpu_victim_cache_admissions = 0;
    uint64_t expert_gpu_victim_cache_filtered_admissions = 0;
    uint64_t expert_gpu_victim_cache_reused_admissions = 0;
    uint64_t expert_gpu_victim_cache_probe_admissions = 0;
    uint64_t expert_gpu_victim_cache_stores = 0;
    uint64_t expert_gpu_victim_cache_evictions = 0;
    uint64_t expert_gpu_victim_cache_dropped_admissions = 0;
    uint64_t expert_gpu_victim_cache_restore_failures = 0;
    uint64_t expert_gpu_victim_cache_bytes_uploaded = 0;
    uint64_t expert_gpu_victim_cache_bytes_downloaded = 0;
    uint64_t expert_gpu_victim_cache_restore_time_microseconds = 0;
    uint64_t expert_gpu_victim_cache_mapped_stores = 0;
    uint64_t expert_gpu_victim_cache_mapped_restores = 0;
    uint64_t expert_gpu_victim_cache_resident_bytes = 0;
    uint64_t expert_gpu_victim_cache_pending_bytes = 0;
    uint64_t expert_gpu_executions = 0;
    uint64_t expert_gpu_execution_failures = 0;
    uint64_t expert_gpu_cpu_preferred = 0;
    uint64_t expert_gpu_execution_time_microseconds = 0;
    uint64_t expert_gpu_arc_recent_bytes = 0;
    uint64_t expert_gpu_arc_frequent_bytes = 0;
    uint64_t expert_gpu_arc_recent_target_bytes = 0;
    uint64_t expert_gpu_arc_recent_ghost_bytes = 0;
    uint64_t expert_gpu_arc_frequent_ghost_bytes = 0;
    uint64_t expert_gpu_device_source_hits = 0;
    uint64_t expert_gpu_device_source_misses = 0;
    uint64_t expert_gpu_device_source_executions = 0;
    uint64_t expert_gpu_device_source_execution_failures = 0;
    uint64_t expert_parallel_tasks = 0;
    uint64_t mxfp4_decode_gemv_rows = 0;
    uint64_t mxfp4_prefill_gemm_rows = 0;
    uint64_t mxfp4_paired_rows = 0;
    uint64_t mxfp4_fused_gate_up_rows = 0;
    uint64_t mxfp4_reused_input_rows = 0;
    uint64_t vulkan_linear_dispatches = 0;
    uint64_t vulkan_attention_blocks = 0;
    uint64_t vulkan_compute_submissions = 0;
    uint64_t vulkan_submit_wait_time_microseconds = 0;
    uint64_t vulkan_batch_uploads = 0;
    uint64_t vulkan_batch_downloads = 0;
    uint64_t vulkan_auxiliary_uploads = 0;
    uint64_t vulkan_auxiliary_upload_bytes = 0;
    uint64_t vulkan_staging_slot_resizes = 0;
    uint64_t vulkan_staging_slot_reuses = 0;
    uint64_t vulkan_staging_slot_acquisitions = 0;
    uint64_t vulkan_staging_slot_contentions = 0;
    uint64_t vulkan_command_buffer_reuses = 0;
    uint64_t vulkan_attention_qkv_rope_fusions = 0;
    uint64_t vulkan_attention_qkv_ring_fusions = 0;
    uint64_t vulkan_attention_decode_sdpa_fusions = 0;
    uint64_t vulkan_kv_ring_appends = 0;
    uint64_t vulkan_kv_ring_resizes = 0;
    uint64_t vulkan_kv_ring_wrapped_views = 0;
    uint64_t attention_time_microseconds = 0;
    uint64_t router_time_microseconds = 0;
    uint64_t expert_time_microseconds = 0;
    uint64_t expert_cache_wait_time_microseconds = 0;
    uint64_t expert_cache_management_time_microseconds = 0;
    uint64_t expert_engine_time_microseconds = 0;
    uint64_t expert_compute_time_microseconds = 0;
    uint64_t expert_regroup_time_microseconds = 0;
    uint64_t expert_combine_time_microseconds = 0;
    uint64_t embedding_time_microseconds = 0;
    uint64_t final_norm_time_microseconds = 0;
    uint64_t lm_head_time_microseconds = 0;
    uint64_t speculative_proposals = 0;
    uint64_t speculative_draft_tokens = 0;
    uint64_t speculative_accepted_tokens = 0;
    uint64_t speculative_context_time_microseconds = 0;
    uint64_t speculative_draft_time_microseconds = 0;
    uint64_t speculative_verify_time_microseconds = 0;
    uint64_t kv_cache_logical_bytes = 0;
    uint64_t kv_cache_allocated_bytes = 0;
    std::vector<uint64_t> expert_token_counts;
};

struct SessionOptions
{
    LogitsOutputMode logits_output_mode = LogitsOutputMode::FullLogits;
    uint64_t sampling_seed = 0;
    uint32_t prefill_chunk_size = 256;
    bool enable_speculative_context = true;
};

class Session
{
private:
    explicit Session(ModelPtr model, const SessionOptions& options);
    [[nodiscard]] Result<PrefillResult> prefill_unlocked(std::span<const int32_t> input_ids);
    [[nodiscard]] Result<DecodeResult> decode_unlocked(int32_t input_id);
    [[nodiscard]] Result<SampledToken> sample_unlocked(std::span<const float> logits, const SamplingOptions& options);
    [[nodiscard]] Result<SampledToken> sample_unlocked(const LogitsOutput& logits, const SamplingOptions& options);

    ModelPtr model_;
    uint64_t sequence_length_ = 0;
    SessionStatistics statistics_;
    SessionStatistics statistics_scratch_;
    std::unique_ptr<CpuSessionState> state_;
    std::unique_ptr<IExecutor> executor_;
    std::mt19937_64 random_generator_;
    uint32_t prefill_chunk_size_ = 256;
    bool speculative_context_enabled_ = true;
    mutable std::mutex mutex_;

    friend class Runtime;
    friend class SessionBatchAccess;

public:
    ~Session();

    [[nodiscard]] Result<PrefillResult> prefill(std::span<const int32_t> input_ids);
    [[nodiscard]] Result<DecodeResult> decode(int32_t input_id);
    [[nodiscard]] Result<SampledToken> sample(const LogitsOutput& logits, const SamplingOptions& options = {});
    [[nodiscard]] Result<GenerationResult> generate(std::span<const int32_t> input_ids, const GenerationOptions& options = {}, TokenStreamCallback on_token = {}, TokenTextDecoder decode_text = {});
    [[nodiscard]] Result<void> reset();

    [[nodiscard]] uint64_t sequence_length() const noexcept
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return sequence_length_;
    }
    [[nodiscard]] SessionStatistics statistics() const
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return statistics_;
    }
    [[nodiscard]] MemoryManagerStatistics memory_statistics() const;
};

using SessionPtr = std::shared_ptr<Session>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SESSION_H
