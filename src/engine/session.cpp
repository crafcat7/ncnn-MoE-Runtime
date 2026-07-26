#include "ncnn/moe/session.h"

#include "engine/cpu_executor.h"
#include "engine/cpu_session_state.h"
#include "engine/session_batch.h"

#include <algorithm>
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

struct LogitOrder
{
    const std::vector<float>& values;

    bool operator()(int32_t left, int32_t right) const
    {
        if (values[left] == values[right])
            return left < right;
        return values[left] > values[right];
    }
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

static Result<std::vector<SamplingCandidate>> sampling_distribution(const LogitsOutput& logits, const SamplingOptions& options)
{
    std::vector<int32_t> token_ids(logits.values.size());
    std::iota(token_ids.begin(), token_ids.end(), 0);
    std::stable_sort(token_ids.begin(), token_ids.end(), LogitOrder{logits.values});

    if (!std::isfinite(logits.values[token_ids.front()]))
        return Error{ErrorCode::InvalidArgument, "sampling requires at least one finite logit"};
    if (options.top_k > 0 && options.top_k < token_ids.size())
        token_ids.resize(options.top_k);

    const float maximum = logits.values[token_ids.front()] / options.temperature;
    std::vector<SamplingCandidate> candidates;
    candidates.reserve(token_ids.size());
    float normalizer = 0.0f;
    for (int32_t token_id : token_ids)
    {
        const float logit = logits.values[token_id];
        if (!std::isfinite(logit))
            continue;
        const float probability = std::exp(logit / options.temperature - maximum);
        candidates.push_back(SamplingCandidate{token_id, probability});
        normalizer += probability;
    }
    if (candidates.empty() || !std::isfinite(normalizer) || normalizer <= 0.0f)
        return Error{ErrorCode::InvalidArgument, "sampling distribution is invalid"};
    for (SamplingCandidate& candidate : candidates)
        candidate.probability /= normalizer;

    if (options.min_p > 0.0f)
    {
        const float threshold = candidates.front().probability * options.min_p;
        size_t count = 0;
        for (const SamplingCandidate& candidate : candidates)
        {
            if (candidate.probability >= threshold)
                candidates[count++] = candidate;
        }
        candidates.resize(count);
    }

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

    normalizer = 0.0f;
    for (const SamplingCandidate& candidate : candidates)
        normalizer += candidate.probability;
    for (SamplingCandidate& candidate : candidates)
        candidate.probability /= normalizer;
    return candidates;
}

static void update_cache_statistics(SessionStatistics& statistics, const CpuSessionState& state)
{
    statistics.kv_cache_logical_bytes = state.kv_cache_logical_bytes();
    statistics.kv_cache_allocated_bytes = state.kv_cache_allocated_bytes();
}

Session::Session(ModelPtr model, const SessionOptions& options)
    : model_(std::move(model)),
      state_(new CpuSessionState(model_->execution_graph())),
      executor_(new CpuExecutor),
      random_generator_(options.sampling_seed),
      prefill_chunk_size_(options.prefill_chunk_size)
{
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
    return prefill_unlocked(input_ids);
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
    return decode_unlocked(input_id);
}

Result<SampledToken> Session::sample_unlocked(const LogitsOutput& logits, const SamplingOptions& options)
{
    if (logits.values.empty())
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
        for (size_t token_id = 0; token_id < logits.values.size(); ++token_id)
        {
            const float logit = logits.values[token_id];
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
    auto candidates = sampling_distribution(logits, options);
    if (!candidates)
        return candidates.error();

    if (candidates.value().size() == 1)
        return SampledToken{candidates.value().front().token_id, 1.0f};

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    const float sample_value = distribution(random_generator_);
    float cumulative_probability = 0.0f;
    for (const SamplingCandidate& candidate : candidates.value())
    {
        cumulative_probability += candidate.probability;
        if (sample_value < cumulative_probability)
            return SampledToken{candidate.token_id, candidate.probability};
    }
    const SamplingCandidate& final_candidate = candidates.value().back();
    return SampledToken{final_candidate.token_id, final_candidate.probability};
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
    auto valid_sampling = validate_sampling_options(options.sampling);
    if (!valid_sampling)
        return valid_sampling.error();

    auto prefill_result = prefill_unlocked(input_ids);
    if (!prefill_result)
        return prefill_result.error();

    LogitsOutput logits = std::move(prefill_result).value().logits;
    GenerationResult result;
    result.tokens.reserve(options.max_new_tokens);
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
    return {};
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
