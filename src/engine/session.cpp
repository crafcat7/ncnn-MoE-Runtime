#include "ncnn/moe/session.h"

#include "kernels/cpu_fast_math.h"
#include "engine/cpu_executor.h"
#include "engine/cpu_session_state.h"
#include "engine/expert_backend.h"
#include "engine/session_batch.h"
#include "kernels/cpu_latent_attention.h"
#include "kernels/cpu_state_cache.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace ncnn {
namespace moe {

struct SamplingCandidate
{
    int32_t token_id = -1;
    float probability = 0.0f;
};

struct SamplingScratch
{
    std::vector<int32_t> token_ids;
    std::array<std::vector<SamplingCandidate>, 2> candidates;
    std::vector<float> residual;
};

struct LogitOrder
{
    std::span<const float> values;

    bool operator()(int32_t left, int32_t right) const
    {
        const bool left_finite = std::isfinite(values[left]);
        const bool right_finite = std::isfinite(values[right]);
        if (left_finite != right_finite)
            return left_finite;
        if (!left_finite)
            return left < right;
        if (values[left] == values[right])
            return left < right;
        return values[left] > values[right];
    }
};

class ScopeExit
{
private:
    std::function<void()> function_;
    bool active_ = true;

public:
    explicit ScopeExit(std::function<void()> function)
        : function_(std::move(function))
    {
    }

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

    ~ScopeExit()
    {
        if (active_)
            function_();
    }
};

class ScopedSessionExpertBackendForeground
{
public:
    explicit ScopedSessionExpertBackendForeground(
        const std::shared_ptr<IExpertExecutionBackend>& backend) noexcept
        : backend_(backend)
    {
        if (backend_)
            backend_->set_foreground_active(true);
    }

    ~ScopedSessionExpertBackendForeground()
    {
        if (backend_)
            backend_->set_foreground_active(false);
    }

    ScopedSessionExpertBackendForeground(const ScopedSessionExpertBackendForeground&) = delete;
    ScopedSessionExpertBackendForeground& operator=(const ScopedSessionExpertBackendForeground&) = delete;

private:
    std::shared_ptr<IExpertExecutionBackend> backend_;
};

static bool contains_token(const std::vector<int32_t>& tokens, int32_t token)
{
    return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

static Result<void> validate_sampling_options(const SamplingOptions& options)
{
    if (!std::isfinite(options.temperature) || options.temperature < 0.0f)
        return Error{ErrorCode::InvalidArgument, "temperature must be finite and non-negative"};
    if (!std::isfinite(options.top_p) || options.top_p <= 0.0f || options.top_p > 1.0f)
        return Error{ErrorCode::InvalidArgument, "top_p must be in the range (0, 1]"};
    if (!std::isfinite(options.min_p) || options.min_p < 0.0f || options.min_p > 1.0f)
        return Error{ErrorCode::InvalidArgument, "min_p must be in the range [0, 1]"};
    return {};
}

static Result<void> sampling_distribution_into(
    std::span<const float> logits,
    const SamplingOptions& options,
    std::vector<SamplingCandidate>& candidates,
    std::vector<int32_t>& token_ids)
{
    candidates.clear();

    float maximum_logit = -std::numeric_limits<float>::infinity();
    for (float logit : logits)
    {
        if (std::isfinite(logit))
            maximum_logit = std::max(maximum_logit, logit);
    }
    if (!std::isfinite(maximum_logit))
        return Error{ErrorCode::InvalidArgument, "sampling requires at least one finite logit"};

    const bool has_top_k = options.top_k > 0 && options.top_k < logits.size();
    const bool needs_order = options.top_p < 1.0f;
    if (has_top_k)
    {
        token_ids.resize(options.top_k);
        std::iota(token_ids.begin(), token_ids.end(), 0);
        std::make_heap(token_ids.begin(), token_ids.end(), LogitOrder{logits});
        for (size_t token_id = options.top_k; token_id < logits.size(); ++token_id)
        {
            const int32_t candidate = static_cast<int32_t>(token_id);
            if (LogitOrder{logits}(candidate, token_ids.front()))
            {
                std::pop_heap(token_ids.begin(), token_ids.end(), LogitOrder{logits});
                token_ids.back() = candidate;
                std::push_heap(token_ids.begin(), token_ids.end(), LogitOrder{logits});
            }
        }
        if (needs_order)
            std::sort(token_ids.begin(), token_ids.end(), LogitOrder{logits});
    }
    else
    {
        token_ids.resize(logits.size());
        std::iota(token_ids.begin(), token_ids.end(), 0);
        if (needs_order)
            std::sort(token_ids.begin(), token_ids.end(), LogitOrder{logits});
    }

    candidates.reserve(token_ids.size());
    const float maximum = maximum_logit / options.temperature;
    float normalizer = 0.0f;
    for (int32_t token_id : token_ids)
    {
        const float logit = logits[token_id];
        if (!std::isfinite(logit))
            continue;
        const float probability = float_approximate_exp(logit / options.temperature - maximum);
        candidates.push_back(SamplingCandidate{token_id, probability});
        normalizer += probability;
    }
    if (candidates.empty() || !std::isfinite(normalizer) || normalizer <= 0.0f)
        return Error{ErrorCode::InvalidArgument, "sampling distribution is invalid"};
    for (SamplingCandidate& candidate : candidates)
        candidate.probability /= normalizer;

    if (options.min_p > 0.0f)
    {
        float maximum_probability = 0.0f;
        for (const SamplingCandidate& candidate : candidates)
            maximum_probability = std::max(maximum_probability, candidate.probability);
        const float threshold = maximum_probability * options.min_p;
        size_t count = 0;
        for (const SamplingCandidate& candidate : candidates)
        {
            if (candidate.probability >= threshold)
                candidates[count++] = candidate;
        }
        candidates.resize(count);
    }

    if (options.top_p < 1.0f)
    {
        float cumulative_probability = 0.0f;
        size_t top_p_count = 0;
        for (const SamplingCandidate& candidate : candidates)
        {
            cumulative_probability += candidate.probability;
            ++top_p_count;
            if (cumulative_probability >= options.top_p)
                break;
        }
        candidates.resize(top_p_count);
    }

    normalizer = 0.0f;
    for (const SamplingCandidate& candidate : candidates)
        normalizer += candidate.probability;
    if (!std::isfinite(normalizer) || normalizer <= 0.0f)
        return Error{ErrorCode::InvalidArgument, "sampling distribution is invalid"};
    for (SamplingCandidate& candidate : candidates)
        candidate.probability /= normalizer;
    return {};
}

static float candidate_probability(
    std::span<const SamplingCandidate> candidates,
    int32_t token_id) noexcept
{
    for (const SamplingCandidate& candidate : candidates)
    {
        if (candidate.token_id == token_id)
            return candidate.probability;
    }
    return 0.0f;
}

static Result<SampledToken> sample_residual_distribution(
    std::span<const SamplingCandidate> target,
    std::span<const SamplingCandidate> draft,
    size_t vocabulary_size,
    std::vector<float>& residual,
    std::mt19937_64& random_generator)
{
    residual.assign(vocabulary_size, 0.0f);
    for (const SamplingCandidate& candidate : target)
        residual[candidate.token_id] = candidate.probability;
    for (const SamplingCandidate& candidate : draft)
    {
        residual[candidate.token_id] = std::max(
            0.0f,
            residual[candidate.token_id] - candidate.probability);
    }
    float normalizer = std::accumulate(
        residual.begin(),
        residual.end(),
        0.0f);
    if (!std::isfinite(normalizer) || normalizer <= 0.0f)
    {
        return Error{
            ErrorCode::InternalError,
            "speculative residual distribution is empty"};
    }
    std::uniform_real_distribution<float> distribution(
        0.0f,
        normalizer);
    const float sample_value = distribution(random_generator);
    float cumulative = 0.0f;
    for (size_t token_id = 0; token_id < residual.size(); ++token_id)
    {
        cumulative += residual[token_id];
        if (sample_value < cumulative)
        {
            return SampledToken{
                static_cast<int32_t>(token_id),
                residual[token_id] / normalizer};
        }
    }
    for (size_t token_id = residual.size(); token_id != 0; --token_id)
    {
        if (residual[token_id - 1] > 0.0f)
        {
            return SampledToken{
                static_cast<int32_t>(token_id - 1),
                residual[token_id - 1] / normalizer};
        }
    }
    return Error{
        ErrorCode::InternalError,
        "speculative residual sampling failed"};
}

static void update_cache_statistics(SessionStatistics& statistics, const CpuSessionState& state)
{
    statistics.kv_cache_logical_bytes = state.kv_cache_logical_bytes();
    statistics.kv_cache_allocated_bytes = state.kv_cache_allocated_bytes();
}

static uint64_t counter_delta(uint64_t current, uint64_t baseline) noexcept
{
    return current >= baseline ? current - baseline : 0;
}

static RuntimeMetricCounters runtime_metric_counters(
    const SessionStatistics& statistics,
    const SessionStatistics* baseline)
{
    const SessionStatistics empty;
    const SessionStatistics& start = baseline == nullptr ? empty : *baseline;
    RuntimeMetricCounters result;
    result.prefill_tokens = counter_delta(statistics.prefill_tokens, start.prefill_tokens);
    result.decode_tokens = counter_delta(statistics.decode_tokens, start.decode_tokens);
    result.expert_cache_hits = counter_delta(statistics.expert_cache_hits, start.expert_cache_hits);
    result.expert_cache_misses = counter_delta(statistics.expert_cache_misses, start.expert_cache_misses);
    result.expert_io_bytes = counter_delta(statistics.expert_cache_bytes_read, start.expert_cache_bytes_read);
    result.expert_compute_time_microseconds = counter_delta(
        statistics.expert_compute_time_microseconds,
        start.expert_compute_time_microseconds);
    result.gpu_submit_count = counter_delta(
        statistics.vulkan_compute_submissions,
        start.vulkan_compute_submissions);
    result.gpu_wait_time_microseconds = counter_delta(
        statistics.vulkan_submit_wait_time_microseconds,
        start.vulkan_submit_wait_time_microseconds);
    result.gpu_kernel_time_microseconds = counter_delta(
        statistics.expert_gpu_execution_time_microseconds,
        start.expert_gpu_execution_time_microseconds);
    result.gpu_kernel_time_available = counter_delta(
                                           statistics.expert_gpu_executions,
                                           start.expert_gpu_executions)
                                       != 0;
    result.vulkan_linear_dispatches = counter_delta(
        statistics.vulkan_linear_dispatches,
        start.vulkan_linear_dispatches);
    result.vulkan_attention_blocks = counter_delta(
        statistics.vulkan_attention_blocks,
        start.vulkan_attention_blocks);
    result.vulkan_batch_uploads = counter_delta(
        statistics.vulkan_batch_uploads,
        start.vulkan_batch_uploads);
    result.vulkan_batch_downloads = counter_delta(
        statistics.vulkan_batch_downloads,
        start.vulkan_batch_downloads);
    result.vulkan_attention_qkv_rope_fusions = counter_delta(
        statistics.vulkan_attention_qkv_rope_fusions,
        start.vulkan_attention_qkv_rope_fusions);
    result.vulkan_attention_device_rope_fusions = counter_delta(
        statistics.vulkan_attention_device_rope_fusions,
        start.vulkan_attention_device_rope_fusions);
    result.vulkan_attention_qkv_ring_fusions = counter_delta(
        statistics.vulkan_attention_qkv_ring_fusions,
        start.vulkan_attention_qkv_ring_fusions);
    result.vulkan_attention_qkv_rope_pipeline_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_pipeline_failures,
        start.vulkan_attention_qkv_rope_pipeline_failures);
    result.vulkan_attention_qkv_rope_shape_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_shape_failures,
        start.vulkan_attention_qkv_rope_shape_failures);
    result.vulkan_attention_qkv_rope_source_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_source_failures,
        start.vulkan_attention_qkv_rope_source_failures);
    result.vulkan_attention_qkv_rope_norm_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_norm_failures,
        start.vulkan_attention_qkv_rope_norm_failures);
    result.vulkan_attention_qkv_rope_ring_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_ring_failures,
        start.vulkan_attention_qkv_rope_ring_failures);
    result.vulkan_attention_qkv_rope_allocation_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_allocation_failures,
        start.vulkan_attention_qkv_rope_allocation_failures);
    result.vulkan_attention_precondition_failures = counter_delta(
        statistics.vulkan_attention_precondition_failures,
        start.vulkan_attention_precondition_failures);
    result.vulkan_attention_staging_failures = counter_delta(
        statistics.vulkan_attention_staging_failures,
        start.vulkan_attention_staging_failures);
    result.vulkan_attention_norm_failures = counter_delta(
        statistics.vulkan_attention_norm_failures,
        start.vulkan_attention_norm_failures);
    result.vulkan_attention_qkv_failures = counter_delta(
        statistics.vulkan_attention_qkv_failures,
        start.vulkan_attention_qkv_failures);
    result.vulkan_attention_cache_failures = counter_delta(
        statistics.vulkan_attention_cache_failures,
        start.vulkan_attention_cache_failures);
    result.vulkan_attention_sdpa_failures = counter_delta(
        statistics.vulkan_attention_sdpa_failures,
        start.vulkan_attention_sdpa_failures);
    result.vulkan_attention_projection_failures = counter_delta(
        statistics.vulkan_attention_projection_failures,
        start.vulkan_attention_projection_failures);
    result.vulkan_attention_output_failures = counter_delta(
        statistics.vulkan_attention_output_failures,
        start.vulkan_attention_output_failures);
    result.vulkan_attention_submit_failures = counter_delta(
        statistics.vulkan_attention_submit_failures,
        start.vulkan_attention_submit_failures);
    result.vulkan_attention_cache_materializations = counter_delta(
        statistics.vulkan_attention_cache_materializations,
        start.vulkan_attention_cache_materializations);
    result.vulkan_attention_cpu_fallbacks = counter_delta(
        statistics.vulkan_attention_cpu_fallbacks,
        start.vulkan_attention_cpu_fallbacks);
    result.vulkan_gated_delta_fusions = counter_delta(
        statistics.vulkan_gated_delta_fusions,
        start.vulkan_gated_delta_fusions);
    result.vulkan_gated_delta_submissions = counter_delta(
        statistics.vulkan_gated_delta_submissions,
        start.vulkan_gated_delta_submissions);
    result.expert_gpu_cache_hits = counter_delta(
        statistics.expert_gpu_cache_hits,
        start.expert_gpu_cache_hits);
    result.expert_gpu_cache_misses = counter_delta(
        statistics.expert_gpu_cache_misses,
        start.expert_gpu_cache_misses);
    result.expert_gpu_cache_admissions = counter_delta(
        statistics.expert_gpu_cache_admissions,
        start.expert_gpu_cache_admissions);
    result.expert_gpu_cache_stores = counter_delta(
        statistics.expert_gpu_cache_stores,
        start.expert_gpu_cache_stores);
    result.expert_gpu_cache_dropped_admissions = counter_delta(
        statistics.expert_gpu_cache_dropped_admissions,
        start.expert_gpu_cache_dropped_admissions);
    result.expert_gpu_cache_resident_bytes = statistics.expert_gpu_cache_resident_bytes;
    result.expert_gpu_cache_pending_bytes = statistics.expert_gpu_cache_pending_bytes;
    result.expert_gpu_executions = counter_delta(
        statistics.expert_gpu_executions,
        start.expert_gpu_executions);
    result.expert_gpu_execution_failures = counter_delta(
        statistics.expert_gpu_execution_failures,
        start.expert_gpu_execution_failures);
    result.expert_gpu_cpu_preferred = counter_delta(
        statistics.expert_gpu_cpu_preferred,
        start.expert_gpu_cpu_preferred);
    result.expert_gpu_route_aggregation_batches = counter_delta(
        statistics.expert_gpu_route_aggregation_batches,
        start.expert_gpu_route_aggregation_batches);
    result.expert_gpu_route_aggregation_routes = counter_delta(
        statistics.expert_gpu_route_aggregation_routes,
        start.expert_gpu_route_aggregation_routes);
    result.expert_gpu_route_aggregation_bytes_saved = counter_delta(
        statistics.expert_gpu_route_aggregation_bytes_saved,
        start.expert_gpu_route_aggregation_bytes_saved);
    result.expert_cache_resident_bytes = statistics.expert_cache_resident_bytes;
    result.kv_cache_logical_bytes = statistics.kv_cache_logical_bytes;
    result.kv_cache_allocated_bytes = statistics.kv_cache_allocated_bytes;
    return result;
}

SessionMetrics Session::metrics_unlocked() const
{
    SessionMetrics result;
    result.generation = runtime_metric_counters(statistics_, &generation_start_statistics_);
    result.cumulative = runtime_metric_counters(statistics_, nullptr);
    result.gpu_available = model_->hybrid_mode() != HybridMode::CpuOnly;
    result.timing.active = generation_active_;
    result.timing.input_tokens = generation_input_tokens_;
    result.timing.output_tokens = generation_output_tokens_;

    uint64_t elapsed_microseconds = generation_elapsed_microseconds_;
    if (generation_active_)
    {
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - generation_started_);
        elapsed_microseconds = static_cast<uint64_t>(elapsed.count());
    }
    result.timing.elapsed_microseconds = elapsed_microseconds;

    uint64_t ttft_microseconds = 0;
    if (generation_has_first_token_)
    {
        const auto ttft = std::chrono::duration_cast<std::chrono::microseconds>(
            generation_first_token_ready_ - generation_started_);
        ttft_microseconds = static_cast<uint64_t>(ttft.count());
        result.timing.ttft_microseconds = ttft_microseconds;
    }

    if (generation_has_first_token_ && generation_output_tokens_ > 1 && elapsed_microseconds > ttft_microseconds)
    {
        const uint64_t decode_elapsed_microseconds = elapsed_microseconds - ttft_microseconds;
        const uint64_t decode_tokens = generation_output_tokens_ - 1;
        result.timing.tpot_microseconds = static_cast<double>(decode_elapsed_microseconds) / static_cast<double>(decode_tokens);
        result.timing.decode_tokens_per_second = static_cast<double>(decode_tokens) * 1000000.0 / static_cast<double>(decode_elapsed_microseconds);
    }
    return result;
}

Session::Session(ModelPtr model, const SessionOptions& options)
    : model_(std::move(model)),
      state_(new CpuSessionState(model_->execution_graph())),
      executor_(new CpuExecutor),
      random_generator_(options.sampling_seed),
      prefill_chunk_size_(options.prefill_chunk_size),
      speculative_context_enabled_(
          options.enable_speculative_context),
      sampling_scratch_(new SamplingScratch)
{
    state_->speculative_context_enabled = speculative_context_enabled_;
    statistics_.expert_token_counts.resize(model_->descriptor().expert_count, 0);
}

Session::~Session() = default;

Result<PrefillResult> Session::prefill_unlocked(std::span<const int32_t> input_ids)
{
    if (input_ids.empty())
        return Error{ErrorCode::InvalidArgument, "prefill requires at least one token"};
    if (input_ids.size() > std::numeric_limits<uint32_t>::max())
        return Error{ErrorCode::InvalidArgument, "prefill token count exceeds uint32 range"};
    const uint32_t max_context_length = model_->descriptor().layers.empty() ? 0 : model_->descriptor().layers[0].attention.max_context_length;
    if (max_context_length > 0 && input_ids.size() > max_context_length - std::min<uint64_t>(sequence_length_, max_context_length))
        return Error{ErrorCode::InvalidArgument, "prefill exceeds the model context length"};

    statistics_scratch_ = statistics_;
    SessionStatistics& updated_statistics = statistics_scratch_;
    std::vector<float> final_logits;
    size_t processed_tokens = 0;
    while (processed_tokens < input_ids.size())
    {
        const size_t remaining_tokens = input_ids.size() - processed_tokens;
        const size_t chunk_size = prefill_chunk_size_ == 0 ? remaining_tokens : std::min<size_t>(remaining_tokens, prefill_chunk_size_);
        const std::span<const int32_t> chunk = input_ids.subspan(processed_tokens, chunk_size);
        auto chunk_logits = executor_->execute(*model_->compiled_, chunk, updated_statistics, *state_, sequence_length_ + processed_tokens);
        if (!chunk_logits)
            return chunk_logits.error();
        auto speculative_context = executor_->update_speculative_context(
            *model_->compiled_,
            updated_statistics,
            *state_);
        if (!speculative_context)
            return speculative_context.error();
        final_logits = std::move(chunk_logits.value().back());
        processed_tokens += chunk_size;
    }

    std::swap(statistics_, statistics_scratch_);
    statistics_.prefill_tokens += input_ids.size();
    sequence_length_ += input_ids.size();
    update_cache_statistics(statistics_, *state_);

    PrefillResult result;
    result.logits.values = std::move(final_logits);
    result.processed_tokens = static_cast<uint32_t>(input_ids.size());
    return result;
}

Result<PrefillResult> Session::prefill(std::span<const int32_t> input_ids)
{
    const std::lock_guard<std::mutex> lock(mutex_);
    auto result = prefill_unlocked(input_ids);
    if (result)
        generation_start_statistics_ = statistics_;
    return result;
}

Result<DecodeResult> Session::decode_unlocked(int32_t input_id)
{
    const uint32_t max_context_length = model_->descriptor().layers.empty() ? 0 : model_->descriptor().layers[0].attention.max_context_length;
    if (max_context_length > 0 && sequence_length_ >= max_context_length)
        return Error{ErrorCode::InvalidArgument, "decode exceeds the model context length"};
    statistics_scratch_ = statistics_;
    SessionStatistics& updated_statistics = statistics_scratch_;
    const std::span<const int32_t> input(&input_id, 1);
    auto all_logits = executor_->execute(*model_->compiled_, input, updated_statistics, *state_, sequence_length_);
    if (!all_logits)
        return all_logits.error();
    auto speculative_context = executor_->update_speculative_context(
        *model_->compiled_,
        updated_statistics,
        *state_);
    if (!speculative_context)
        return speculative_context.error();

    std::swap(statistics_, statistics_scratch_);
    ++statistics_.decode_tokens;
    ++sequence_length_;
    update_cache_statistics(statistics_, *state_);

    DecodeResult result;
    result.logits.values = std::move(all_logits.value().front());
    result.sequence_length = sequence_length_;
    return result;
}

Result<DecodeResult> Session::decode(int32_t input_id)
{
    const std::lock_guard<std::mutex> lock(mutex_);
    auto result = decode_unlocked(input_id);
    if (result)
        generation_start_statistics_ = statistics_;
    return result;
}

Result<SampledToken> Session::sample_unlocked(
    std::span<const float> logits,
    const SamplingOptions& options)
{
    if (logits.empty())
    {
        return Error{ErrorCode::InvalidArgument, "cannot sample empty logits"};
    }
    auto valid_options = validate_sampling_options(options);
    if (!valid_options)
        return valid_options.error();
    if (options.temperature == 0.0f)
    {
        int32_t selected_token = -1;
        float selected_logit = -std::numeric_limits<float>::infinity();
        for (size_t token_id = 0; token_id < logits.size(); ++token_id)
        {
            const float logit = logits[token_id];
            if (!std::isfinite(logit))
                continue;
            if (selected_token < 0 || logit > selected_logit)
            {
                selected_token = static_cast<int32_t>(token_id);
                selected_logit = logit;
            }
        }
        if (selected_token < 0)
        {
            return Error{ErrorCode::InvalidArgument, "sampling requires at least one finite logit"};
        }
        return SampledToken{selected_token, 1.0f};
    }
    SamplingScratch& scratch = *sampling_scratch_;
    auto distribution_result = sampling_distribution_into(
        logits,
        options,
        scratch.candidates[0],
        scratch.token_ids);
    if (!distribution_result)
        return distribution_result.error();
    const std::vector<SamplingCandidate>& candidates = scratch.candidates[0];

    if (candidates.size() == 1)
        return SampledToken{candidates.front().token_id, 1.0f};

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    const float sample_value = distribution(random_generator_);
    float cumulative_probability = 0.0f;
    for (const SamplingCandidate& candidate : candidates)
    {
        cumulative_probability += candidate.probability;
        if (sample_value < cumulative_probability)
            return SampledToken{candidate.token_id, candidate.probability};
    }
    const SamplingCandidate& final_candidate = candidates.back();
    return SampledToken{final_candidate.token_id, final_candidate.probability};
}

Result<SampledToken> Session::sample_unlocked(
    const LogitsOutput& logits,
    const SamplingOptions& options)
{
    return sample_unlocked(logits.values, options);
}

Result<SampledToken> Session::sample(const LogitsOutput& logits, const SamplingOptions& options)
{
    const std::lock_guard<std::mutex> lock(mutex_);
    return sample_unlocked(logits, options);
}

Result<GenerationResult> Session::generate(std::span<const int32_t> input_ids, const GenerationOptions& options, TokenStreamCallback on_token, TokenTextDecoder decode_text)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (options.max_new_tokens == 0)
        return Error{ErrorCode::InvalidArgument, "max_new_tokens must be non-zero"};
    if (!std::isfinite(options.speculative_confidence_threshold)
        || options.speculative_confidence_threshold < 0.0f
        || options.speculative_confidence_threshold > 1.0f)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "speculative_confidence_threshold must be in the range [0, 1]"};
    }
    auto valid_sampling = validate_sampling_options(options.sampling);
    if (!valid_sampling)
        return valid_sampling.error();

    // Keep device-weight admission out of the foreground generation path.
    const ScopedSessionExpertBackendForeground expert_backend_foreground(
        model_->compiled_->expert_backend);

    generation_active_ = true;
    generation_input_tokens_ = input_ids.size();
    generation_output_tokens_ = 0;
    generation_elapsed_microseconds_ = 0;
    generation_has_first_token_ = false;
    generation_started_ = std::chrono::steady_clock::now();
    generation_first_token_ready_ = {};
    generation_start_statistics_ = statistics_;
    const auto finish_generation_metrics = [this]() {
        if (!generation_active_)
            return;
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - generation_started_);
        generation_elapsed_microseconds_ = static_cast<uint64_t>(elapsed.count());
        generation_active_ = false;
    };
    const ScopeExit generation_metrics_scope(finish_generation_metrics);
    const auto mark_token_ready = [this]() {
        if (!generation_has_first_token_)
        {
            generation_has_first_token_ = true;
            generation_first_token_ready_ = std::chrono::steady_clock::now();
        }
        ++generation_output_tokens_;
    };

    const bool enable_speculative_context = speculative_context_enabled_
                                            && options.enable_speculative
                                            && model_->compiled_->speculative.enabled();
    if (enable_speculative_context
        && !state_->speculative_context_enabled
        && sequence_length_ != 0)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "cannot enable speculative context after non-speculative decoding"};
    }
    state_->speculative_context_enabled = enable_speculative_context;
    auto prefill_result = prefill_unlocked(input_ids);
    if (!prefill_result)
        return prefill_result.error();

    LogitsOutput logits = std::move(prefill_result).value().logits;
    GenerationResult result;
    result.tokens.reserve(options.max_new_tokens);
    if (enable_speculative_context)
    {
        const bool state_cache_transactions = model_->compiled_->speculative.kind
                                              == SpeculativeModelKind::Mtp;
        const auto begin_cache_transaction =
            [state_cache_transactions](
                std::span<CpuLayerCache> caches,
                size_t expected_rows) -> Result<void> {
            if (state_cache_transactions)
            {
                return begin_state_cache_transaction(
                    caches,
                    expected_rows);
            }
            begin_latent_cache_transaction(caches);
            return {};
        };
        const auto finish_cache_transaction =
            [state_cache_transactions](
                std::span<CpuLayerCache> caches,
                size_t committed_rows) -> Result<void> {
            return state_cache_transactions
                       ? finish_state_cache_transaction(
                             caches,
                             committed_rows)
                       : finish_latent_cache_transaction(
                             caches,
                             committed_rows);
        };
        auto initial = sample_unlocked(logits, options.sampling);
        if (!initial)
            return initial.error();
        StreamToken initial_token;
        initial_token.index = 0;
        initial_token.token_id = initial.value().token_id;
        initial_token.probability = initial.value().probability;
        initial_token.is_stop_token = contains_token(options.stop_tokens, initial_token.token_id);
        const uint64_t initial_sequence_length = sequence_length_;
        mark_token_ready();
        lock.unlock();
        if (decode_text)
            initial_token.text = decode_text(initial_token.token_id);
        const bool initial_continue = !on_token || on_token(initial_token);
        lock.lock();
        if (sequence_length_ != initial_sequence_length)
        {
            return Error{
                ErrorCode::InvalidArgument,
                "generation callback modified the Session"};
        }
        result.tokens.push_back(std::move(initial_token));
        if (!initial_continue)
        {
            result.stopped_by_callback = true;
            return result;
        }
        if (result.tokens.back().is_stop_token)
        {
            result.stopped_by_stop_token = true;
            return result;
        }

        bool speculative_active = true;
        while (result.tokens.size() < options.max_new_tokens)
        {
            const int32_t anchor = result.tokens.back().token_id;
            const size_t remaining = options.max_new_tokens - result.tokens.size();
            if (remaining == 1 || !speculative_active)
            {
                auto decoded = decode_unlocked(anchor);
                if (!decoded)
                    return decoded.error();
                auto sampled = sample_unlocked(
                    decoded.value().logits,
                    options.sampling);
                if (!sampled)
                    return sampled.error();
                StreamToken token;
                token.index = static_cast<uint32_t>(result.tokens.size());
                token.token_id = sampled.value().token_id;
                token.probability = sampled.value().probability;
                token.is_stop_token = contains_token(options.stop_tokens, token.token_id);
                const uint64_t expected_sequence_length = sequence_length_;
                mark_token_ready();
                lock.unlock();
                if (decode_text)
                    token.text = decode_text(token.token_id);
                const bool callback_continue = !on_token || on_token(token);
                lock.lock();
                if (sequence_length_ != expected_sequence_length)
                {
                    return Error{
                        ErrorCode::InvalidArgument,
                        "generation callback modified the Session"};
                }
                result.tokens.push_back(std::move(token));
                if (!callback_continue)
                {
                    result.stopped_by_callback = true;
                    break;
                }
                if (result.tokens.back().is_stop_token)
                {
                    result.stopped_by_stop_token = true;
                    break;
                }
                continue;
            }
            auto draft_transaction = begin_cache_transaction(
                state_->speculative_layers,
                model_->compiled_->speculative.block_size);
            if (!draft_transaction)
                return draft_transaction.error();
            auto proposed = executor_->propose_speculative(
                *model_->compiled_,
                anchor,
                statistics_,
                *state_,
                sequence_length_,
                [this, &options](
                    const std::vector<float>& draft_logits)
                    -> Result<int32_t> {
                    auto sampled = sample_unlocked(
                        draft_logits,
                        options.sampling);
                    if (!sampled)
                        return sampled.error();
                    return sampled.value().token_id;
                });
            auto discarded_draft_cache = finish_cache_transaction(
                state_->speculative_layers,
                proposed
                    ? proposed.value().committed_context_rows
                    : 0);
            if (!discarded_draft_cache)
                return discarded_draft_cache.error();
            if (!proposed)
                return proposed.error();
            size_t draft_count = std::min(
                proposed.value().token_ids.size(),
                remaining - 1);
            if (options.speculative_max_draft_tokens != 0)
            {
                draft_count = std::min(
                    draft_count,
                    static_cast<size_t>(
                        options.speculative_max_draft_tokens));
            }
            if (options.speculative_confidence_threshold > 0.0f
                && !proposed.value().confidence_logits.empty())
            {
                draft_count = std::min(
                    draft_count,
                    proposed.value().confidence_logits.size());
                for (size_t index = 0; index < draft_count; ++index)
                {
                    const float confidence = 1.0f
                                             / (1.0f
                                                + float_approximate_exp(
                                                    -proposed.value().confidence_logits[index]));
                    if (confidence
                        < options.speculative_confidence_threshold)
                    {
                        draft_count = index;
                        break;
                    }
                }
            }
            if (draft_count == 0)
            {
                speculative_active = false;
                continue;
            }
            std::vector<int32_t> verify_input_ids;
            verify_input_ids.reserve(draft_count + 1);
            verify_input_ids.push_back(anchor);
            verify_input_ids.insert(
                verify_input_ids.end(),
                proposed.value().token_ids.begin(),
                proposed.value().token_ids.begin() + draft_count);

            statistics_scratch_ = statistics_;
            auto target_transaction = begin_cache_transaction(
                state_->layers,
                verify_input_ids.size());
            if (!target_transaction)
                return target_transaction.error();
            const auto verify_started = std::chrono::steady_clock::now();
            auto execute_target_verify =
                [&]() -> Result<std::vector<std::vector<float>>> {
                if (!state_cache_transactions)
                {
                    return executor_->execute(
                        *model_->compiled_,
                        verify_input_ids,
                        statistics_scratch_,
                        *state_,
                        sequence_length_);
                }

                std::vector<std::vector<float>> logits;
                logits.reserve(verify_input_ids.size());
                CpuBatch verified_hidden(
                    verify_input_ids.size(),
                    model_->compiled_->descriptor.hidden_size);
                for (size_t index = 0;
                     index < verify_input_ids.size();
                     ++index)
                {
                    const std::span<const int32_t> input(
                        &verify_input_ids[index],
                        1);
                    auto row_logits = executor_->execute(
                        *model_->compiled_,
                        input,
                        statistics_scratch_,
                        *state_,
                        sequence_length_ + index);
                    if (!row_logits)
                        return row_logits.error();
                    std::vector<std::vector<float>> rows = std::move(row_logits).value();
                    if (rows.size() != 1
                        || state_->speculative_main_hidden.rows() != 1)
                    {
                        return Error{
                            ErrorCode::InternalError,
                            "sequential MTP verification produced invalid rows"};
                    }
                    logits.push_back(std::move(rows.front()));
                    std::copy_n(
                        state_->speculative_main_hidden.row(0),
                        model_->compiled_->descriptor.hidden_size,
                        verified_hidden.row(index));
                }
                state_->speculative_main_hidden = std::move(verified_hidden);
                state_->speculative_main_hidden_position = sequence_length_;
                return logits;
            };
            auto target_logits = execute_target_verify();
            if (!target_logits)
            {
                auto rolled_back = finish_cache_transaction(
                    state_->layers,
                    0);
                if (!rolled_back)
                    return rolled_back.error();
                return target_logits.error();
            }
            const auto target_verify_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - verify_started);
            const uint64_t target_verify_microseconds = static_cast<uint64_t>(target_verify_elapsed.count());
            size_t accepted = 0;
            std::vector<float> accepted_probabilities;
            accepted_probabilities.reserve(draft_count);
            std::vector<SamplingCandidate> rejected_target_distribution;
            std::vector<SamplingCandidate> rejected_draft_distribution;
            while (accepted < draft_count)
            {
                if (options.sampling.temperature == 0.0f)
                {
                    auto target_token = sample_unlocked(
                        target_logits.value()[accepted],
                        options.sampling);
                    if (!target_token)
                    {
                        auto rolled_back = finish_cache_transaction(
                            state_->layers,
                            0);
                        if (!rolled_back)
                            return rolled_back.error();
                        return target_token.error();
                    }
                    if (target_token.value().token_id
                        != proposed.value().token_ids[accepted])
                    {
                        break;
                    }
                    accepted_probabilities.push_back(1.0f);
                }
                else
                {
                    SamplingScratch& sampling_scratch = *sampling_scratch_;
                    auto target_distribution = sampling_distribution_into(
                        target_logits.value()[accepted],
                        options.sampling,
                        sampling_scratch.candidates[0],
                        sampling_scratch.token_ids);
                    auto draft_distribution = sampling_distribution_into(
                        proposed.value().logits[accepted],
                        options.sampling,
                        sampling_scratch.candidates[1],
                        sampling_scratch.token_ids);
                    if (!target_distribution || !draft_distribution)
                    {
                        auto rolled_back = finish_cache_transaction(
                            state_->layers,
                            0);
                        if (!rolled_back)
                            return rolled_back.error();
                        return !target_distribution
                                   ? target_distribution.error()
                                   : draft_distribution.error();
                    }
                    const int32_t token_id = proposed.value().token_ids[accepted];
                    const float target_probability = candidate_probability(
                        sampling_scratch.candidates[0],
                        token_id);
                    const float draft_probability = candidate_probability(
                        sampling_scratch.candidates[1],
                        token_id);
                    if (draft_probability <= 0.0f)
                    {
                        auto rolled_back = finish_cache_transaction(
                            state_->layers,
                            0);
                        if (!rolled_back)
                            return rolled_back.error();
                        return Error{
                            ErrorCode::InternalError,
                            "sampled draft token has zero probability"};
                    }
                    const float acceptance_probability = std::min(
                        1.0f,
                        target_probability / draft_probability);
                    std::uniform_real_distribution<float> distribution(
                        0.0f,
                        1.0f);
                    if (distribution(random_generator_)
                        >= acceptance_probability)
                    {
                        rejected_target_distribution.assign(
                            sampling_scratch.candidates[0].begin(),
                            sampling_scratch.candidates[0].end());
                        rejected_draft_distribution.assign(
                            sampling_scratch.candidates[1].begin(),
                            sampling_scratch.candidates[1].end());
                        break;
                    }
                    accepted_probabilities.push_back(
                        target_probability);
                }
                ++accepted;
            }
            if (accepted == 0 && !state_cache_transactions)
            {
                auto rolled_back = finish_cache_transaction(
                    state_->layers,
                    0);
                if (!rolled_back)
                    return rolled_back.error();
                statistics_scratch_.speculative_verify_time_microseconds += target_verify_microseconds;
                std::swap(statistics_, statistics_scratch_);
                update_cache_statistics(statistics_, *state_);
                speculative_active = false;
                continue;
            }
            auto next = [&]() -> Result<SampledToken> {
                if (accepted < draft_count
                    && options.sampling.temperature > 0.0f)
                {
                    return sample_residual_distribution(
                        rejected_target_distribution,
                        rejected_draft_distribution,
                        target_logits.value()[accepted].size(),
                        sampling_scratch_->residual,
                        random_generator_);
                }
                return sample_unlocked(
                    target_logits.value()[accepted],
                    options.sampling);
            }();
            if (!next)
            {
                auto rolled_back = finish_cache_transaction(
                    state_->layers,
                    0);
                if (!rolled_back)
                    return rolled_back.error();
                return next.error();
            }

            std::vector<SampledToken> output_tokens;
            output_tokens.reserve(accepted + 1);
            for (size_t index = 0; index < accepted; ++index)
            {
                output_tokens.push_back(
                    {proposed.value().token_ids[index],
                     accepted_probabilities[index]});
            }
            output_tokens.push_back(next.value());
            size_t emitted = 0;
            bool continue_generation = true;
            for (const SampledToken& sampled : output_tokens)
            {
                StreamToken token;
                token.index = static_cast<uint32_t>(result.tokens.size());
                token.token_id = sampled.token_id;
                token.probability = sampled.probability;
                token.is_stop_token = contains_token(options.stop_tokens, token.token_id);
                const uint64_t expected_sequence_length = sequence_length_;
                mark_token_ready();
                lock.unlock();
                if (decode_text)
                    token.text = decode_text(token.token_id);
                const bool callback_continue = !on_token || on_token(token);
                lock.lock();
                if (sequence_length_ != expected_sequence_length)
                {
                    auto rolled_back = finish_cache_transaction(
                        state_->layers,
                        0);
                    if (!rolled_back)
                        return rolled_back.error();
                    return Error{
                        ErrorCode::InvalidArgument,
                        "generation callback modified the Session"};
                }
                result.tokens.push_back(std::move(token));
                ++emitted;
                if (!callback_continue)
                {
                    result.stopped_by_callback = true;
                    continue_generation = false;
                    break;
                }
                if (result.tokens.back().is_stop_token)
                {
                    result.stopped_by_stop_token = true;
                    continue_generation = false;
                    break;
                }
                if (result.tokens.size() == options.max_new_tokens)
                {
                    continue_generation = false;
                    break;
                }
            }

            auto committed = finish_cache_transaction(
                state_->layers,
                emitted);
            if (!committed)
            {
                return committed.error();
            }
            state_->speculative_main_hidden.reset(
                emitted,
                state_->speculative_main_hidden.columns(),
                false);
            if (state_cache_transactions)
            {
                state_->speculative_direct_alignment_ids.resize(
                    emitted);
                for (size_t index = 0; index < emitted; ++index)
                {
                    state_->speculative_direct_alignment_ids[index] = output_tokens[index].token_id;
                }
            }
            auto speculative_context = executor_->update_speculative_context(
                *model_->compiled_,
                statistics_scratch_,
                *state_);
            if (!speculative_context)
                return speculative_context.error();
            statistics_scratch_.decode_tokens += emitted;
            statistics_scratch_.speculative_accepted_tokens += emitted > 0 ? emitted - 1 : 0;
            const auto verify_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - verify_started);
            statistics_scratch_.speculative_verify_time_microseconds += static_cast<uint64_t>(verify_elapsed.count());
            std::swap(statistics_, statistics_scratch_);
            sequence_length_ += emitted;
            update_cache_statistics(statistics_, *state_);
            if (!continue_generation)
                break;
        }
        return result;
    }
    for (uint32_t index = 0; index < options.max_new_tokens; ++index)
    {
        auto sampled = sample_unlocked(logits, options.sampling);
        if (!sampled)
            return sampled.error();

        StreamToken token;
        token.index = index;
        token.token_id = sampled.value().token_id;
        token.probability = sampled.value().probability;
        token.is_stop_token = contains_token(options.stop_tokens, token.token_id);

        const uint64_t expected_sequence_length = sequence_length_;
        mark_token_ready();
        lock.unlock();
        if (decode_text)
            token.text = decode_text(token.token_id);
        const bool continue_generation = !on_token || on_token(token);
        lock.lock();
        if (sequence_length_ != expected_sequence_length)
        {
            return Error{ErrorCode::InvalidArgument, "generation callback modified the Session"};
        }

        result.tokens.push_back(std::move(token));
        if (!continue_generation)
        {
            result.stopped_by_callback = true;
            break;
        }
        if (result.tokens.back().is_stop_token)
        {
            result.stopped_by_stop_token = true;
            break;
        }
        if (index + 1 == options.max_new_tokens)
            break;

        auto decoded = decode_unlocked(token.token_id);
        if (!decoded)
            return decoded.error();
        logits = std::move(decoded).value().logits;
    }
    return result;
}

Result<void> Session::reset()
{
    const std::lock_guard<std::mutex> lock(mutex_);
    sequence_length_ = 0;
    statistics_ = {};
    statistics_.expert_token_counts.resize(model_->descriptor().expert_count, 0);
    statistics_scratch_ = {};
    statistics_scratch_.expert_token_counts.resize(model_->descriptor().expert_count, 0);
    state_.reset(new CpuSessionState(model_->execution_graph()));
    state_->speculative_context_enabled = speculative_context_enabled_;
    generation_start_statistics_ = {};
    generation_active_ = false;
    generation_input_tokens_ = 0;
    generation_output_tokens_ = 0;
    generation_elapsed_microseconds_ = 0;
    generation_has_first_token_ = false;
    generation_started_ = {};
    generation_first_token_ready_ = {};
    return {};
}

SessionMetrics Session::metrics() const
{
    const std::lock_guard<std::mutex> lock(mutex_);
    return metrics_unlocked();
}

MemoryManagerStatistics Session::memory_statistics() const
{
    const std::lock_guard<std::mutex> lock(mutex_);
    return state_->memory_manager.statistics();
}

bool SessionBatchAccess::compatible(std::span<Session* const> sessions) noexcept
{
    if (sessions.size() < 2 || !sessions.front())
        return false;
    const Model* model = sessions.front()->model_.get();
    for (const Session* session : sessions)
    {
        if (!session || session->model_.get() != model)
            return false;
    }
    return true;
}

Result<std::vector<PrefillResult>> SessionBatchAccess::prefill(
    std::span<Session* const> sessions,
    std::span<const std::vector<int32_t>> input_ids,
    StagedDecodeBatchMetrics& metrics)
{
    if (sessions.empty() || sessions.size() != input_ids.size())
    {
        return Error{
            ErrorCode::InvalidArgument,
            "staged prefill requires one input sequence per session"};
    }

    std::vector<Session*> lock_order(sessions.begin(), sessions.end());
    for (const Session* session : lock_order)
    {
        if (!session)
        {
            return Error{
                ErrorCode::InvalidArgument,
                "staged prefill session cannot be null"};
        }
    }
    std::sort(lock_order.begin(), lock_order.end(), std::less<Session*>());
    if (std::adjacent_find(lock_order.begin(), lock_order.end())
        != lock_order.end())
    {
        return Error{
            ErrorCode::InvalidArgument,
            "staged prefill requires unique sessions"};
    }
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(lock_order.size());
    for (Session* session : lock_order)
        locks.emplace_back(session->mutex_);

    const ModelPtr& model = sessions.front()->model_;
    size_t maximum_tokens = 0;
    for (size_t index = 0; index < sessions.size(); ++index)
    {
        Session& session = *sessions[index];
        if (session.model_.get() != model.get())
        {
            return Error{
                ErrorCode::InvalidArgument,
                "staged prefill sessions must share one loaded model"};
        }
        if (input_ids[index].empty())
        {
            return Error{
                ErrorCode::InvalidArgument,
                "staged prefill requires non-empty input sequences"};
        }
        if (input_ids[index].size()
            > std::numeric_limits<uint32_t>::max())
        {
            return Error{
                ErrorCode::InvalidArgument,
                "prefill token count exceeds uint32 range"};
        }
        const uint32_t max_context_length = model->descriptor().layers.empty()
                                                ? 0
                                                : model->descriptor().layers[0].attention.max_context_length;
        if (max_context_length > 0
            && input_ids[index].size()
                   > max_context_length
                         - std::min<uint64_t>(
                             session.sequence_length_,
                             max_context_length))
        {
            return Error{
                ErrorCode::InvalidArgument,
                "prefill exceeds the model context length"};
        }
        session.statistics_scratch_ = session.statistics_;
        maximum_tokens = std::max(maximum_tokens, input_ids[index].size());
    }

    CpuExecutor executor;
    std::vector<std::vector<float>> final_logits(sessions.size());
    uint64_t expert_time_before = 0;
    uint64_t expert_cache_wait_before = 0;
    for (Session* session : sessions)
    {
        expert_time_before += session->statistics_.expert_time_microseconds;
        expert_cache_wait_before += session->statistics_.expert_cache_wait_time_microseconds;
    }

    for (size_t token_index = 0;
         token_index < maximum_tokens;
         ++token_index)
    {
        std::vector<size_t> active_indices;
        std::vector<CpuDecodeBatchEntry> entries;
        active_indices.reserve(sessions.size());
        entries.reserve(sessions.size());
        for (size_t session_index = 0;
             session_index < sessions.size();
             ++session_index)
        {
            if (token_index >= input_ids[session_index].size())
                continue;
            Session& session = *sessions[session_index];
            active_indices.push_back(session_index);
            entries.push_back({
                input_ids[session_index][token_index],
                &session.statistics_scratch_,
                session.state_.get(),
                session.sequence_length_ + token_index,
            });
        }

        CpuDecodeBatchMetrics execution_metrics;
        auto logits = executor.execute_decode_batch(
            *model->compiled_,
            entries,
            execution_metrics);
        if (!logits)
            return logits.error();
        if (logits.value().size() != active_indices.size())
        {
            return Error{
                ErrorCode::InternalError,
                "staged prefill returned an invalid result count"};
        }
        metrics.logical_expert_batches += execution_metrics.logical_expert_batches;
        metrics.physical_expert_batches += execution_metrics.physical_expert_batches;
        metrics.coalesced_expert_routes += execution_metrics.coalesced_expert_routes;
        metrics.vulkan_attention_batch_submissions += execution_metrics.vulkan_attention_batch_submissions;
        metrics.vulkan_attention_batch_rows += execution_metrics.vulkan_attention_batch_rows;
        metrics.vulkan_attention_batch_avoided_submissions += execution_metrics.vulkan_attention_batch_avoided_submissions;
        metrics.max_expert_batch_size = std::max(
            metrics.max_expert_batch_size,
            execution_metrics.max_expert_batch_size);
        for (size_t active_index = 0;
             active_index < active_indices.size();
             ++active_index)
        {
            const size_t session_index = active_indices[active_index];
            Session& session = *sessions[session_index];
            auto speculative_context = executor.update_speculative_context(
                *model->compiled_,
                session.statistics_scratch_,
                *session.state_);
            if (!speculative_context)
                return speculative_context.error();
            final_logits[session_index] = std::move(logits.value()[active_index]);
        }
    }

    std::vector<PrefillResult> results(sessions.size());
    for (size_t index = 0; index < sessions.size(); ++index)
    {
        Session& session = *sessions[index];
        std::swap(
            session.statistics_,
            session.statistics_scratch_);
        session.statistics_.prefill_tokens += input_ids[index].size();
        session.sequence_length_ += input_ids[index].size();
        update_cache_statistics(
            session.statistics_,
            *session.state_);
        results[index].logits.values = std::move(final_logits[index]);
        results[index].processed_tokens = static_cast<uint32_t>(input_ids[index].size());
    }
    for (Session* session : sessions)
    {
        metrics.expert_time_microseconds += session->statistics_.expert_time_microseconds;
        metrics.expert_cache_wait_time_microseconds += session->statistics_.expert_cache_wait_time_microseconds;
    }
    metrics.expert_time_microseconds -= expert_time_before;
    metrics.expert_cache_wait_time_microseconds -= expert_cache_wait_before;
    return results;
}

Result<std::vector<DecodeResult>> SessionBatchAccess::decode(std::span<Session* const> sessions, std::span<const int32_t> input_ids, StagedDecodeBatchMetrics& metrics)
{
    if (sessions.empty() || sessions.size() != input_ids.size())
    {
        return Error{ErrorCode::InvalidArgument, "staged decode requires one input id per session"};
    }

    std::vector<Session*> lock_order(sessions.begin(), sessions.end());
    for (const Session* session : lock_order)
    {
        if (!session)
        {
            return Error{ErrorCode::InvalidArgument, "staged decode session cannot be null"};
        }
    }
    std::sort(lock_order.begin(), lock_order.end(), std::less<Session*>());
    if (std::adjacent_find(lock_order.begin(), lock_order.end()) != lock_order.end())
    {
        return Error{ErrorCode::InvalidArgument, "staged decode requires unique sessions"};
    }
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(lock_order.size());
    for (Session* session : lock_order)
        locks.emplace_back(session->mutex_);

    const ModelPtr& model = sessions.front()->model_;
    for (size_t index = 0; index < sessions.size(); ++index)
    {
        Session& session = *sessions[index];
        if (session.model_.get() != model.get())
        {
            return Error{ErrorCode::InvalidArgument, "staged decode sessions must share one loaded model"};
        }
        const uint32_t max_context_length = model->descriptor().layers.empty() ? 0 : model->descriptor().layers[0].attention.max_context_length;
        if (max_context_length > 0 && session.sequence_length_ >= max_context_length)
        {
            return Error{ErrorCode::InvalidArgument, "decode exceeds the model context length"};
        }
    }

    std::vector<CpuDecodeBatchEntry> entries(sessions.size());
    uint64_t expert_time_before = 0;
    uint64_t expert_cache_wait_before = 0;
    for (size_t index = 0; index < sessions.size(); ++index)
    {
        Session& session = *sessions[index];
        expert_time_before += session.statistics_.expert_time_microseconds;
        expert_cache_wait_before += session.statistics_.expert_cache_wait_time_microseconds;
        session.statistics_scratch_ = session.statistics_;
        entries[index] = {
            input_ids[index],
            &session.statistics_scratch_,
            session.state_.get(),
            session.sequence_length_,
        };
    }

    CpuDecodeBatchMetrics execution_metrics;
    CpuExecutor executor;
    auto logits = executor.execute_decode_batch(*model->compiled_, entries, execution_metrics);
    if (!logits)
        return logits.error();
    if (logits.value().size() != sessions.size())
    {
        return Error{ErrorCode::InternalError, "staged decode returned an invalid result count"};
    }
    for (size_t index = 0; index < sessions.size(); ++index)
    {
        auto speculative_context = executor.update_speculative_context(
            *model->compiled_,
            sessions[index]->statistics_scratch_,
            *sessions[index]->state_);
        if (!speculative_context)
            return speculative_context.error();
    }

    std::vector<DecodeResult> results(sessions.size());
    for (size_t index = 0; index < sessions.size(); ++index)
    {
        Session& session = *sessions[index];
        std::swap(session.statistics_, session.statistics_scratch_);
        ++session.statistics_.decode_tokens;
        ++session.sequence_length_;
        update_cache_statistics(session.statistics_, *session.state_);
        results[index].logits.values = std::move(logits.value()[index]);
        results[index].sequence_length = session.sequence_length_;
    }
    for (Session* session : sessions)
    {
        metrics.expert_time_microseconds += session->statistics_.expert_time_microseconds;
        metrics.expert_cache_wait_time_microseconds += session->statistics_.expert_cache_wait_time_microseconds;
    }
    metrics.expert_time_microseconds -= expert_time_before;
    metrics.expert_cache_wait_time_microseconds -= expert_cache_wait_before;
    metrics.logical_expert_batches += execution_metrics.logical_expert_batches;
    metrics.physical_expert_batches += execution_metrics.physical_expert_batches;
    metrics.coalesced_expert_routes += execution_metrics.coalesced_expert_routes;
    metrics.vulkan_attention_batch_submissions += execution_metrics.vulkan_attention_batch_submissions;
    metrics.vulkan_attention_batch_rows += execution_metrics.vulkan_attention_batch_rows;
    metrics.vulkan_attention_batch_avoided_submissions += execution_metrics.vulkan_attention_batch_avoided_submissions;
    metrics.max_expert_batch_size = std::max(metrics.max_expert_batch_size, execution_metrics.max_expert_batch_size);
    return results;
}

SessionDecodePhaseSnapshot SessionBatchAccess::phase_snapshot(Session& session)
{
    const std::lock_guard<std::mutex> lock(session.mutex_);
    return {
        session.statistics_.expert_time_microseconds,
        session.statistics_.expert_cache_wait_time_microseconds,
    };
}

} // namespace moe
} // namespace ncnn
