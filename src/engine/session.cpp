#include "ncnn/moe/session.h"

#include "engine/cpu_executor.h"
#include "engine/cpu_session_state.h"
#include "engine/session_batch.h"
#include "kernels/cpu_latent_attention.h"

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

struct LogitOrder
{
    std::span<const float> values;

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

static Result<std::vector<SamplingCandidate>> sampling_distribution(
    std::span<const float> logits,
    const SamplingOptions& options)
{
    std::vector<int32_t> token_ids(logits.size());
    std::iota(token_ids.begin(), token_ids.end(), 0);
    std::stable_sort(token_ids.begin(), token_ids.end(), LogitOrder{logits});

    if (!std::isfinite(logits[token_ids.front()]))
        return Error{ErrorCode::InvalidArgument, "sampling requires at least one finite logit"};
    if (options.top_k > 0 && options.top_k < token_ids.size())
        token_ids.resize(options.top_k);

    const float maximum = logits[token_ids.front()] / options.temperature;
    std::vector<SamplingCandidate> candidates;
    candidates.reserve(token_ids.size());
    float normalizer = 0.0f;
    for (int32_t token_id : token_ids)
    {
        const float logit = logits[token_id];
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
    std::mt19937_64& random_generator)
{
    std::vector<float> residual(vocabulary_size, 0.0f);
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
    return decode_unlocked(input_id);
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

    auto prefill_result = prefill_unlocked(input_ids);
    if (!prefill_result)
        return prefill_result.error();

    LogitsOutput logits = std::move(prefill_result).value().logits;
    GenerationResult result;
    result.tokens.reserve(options.max_new_tokens);
    if (options.enable_speculative
        && model_->compiled_->speculative.enabled())
    {
        auto initial = sample_unlocked(logits, options.sampling);
        if (!initial)
            return initial.error();
        StreamToken initial_token;
        initial_token.index = 0;
        initial_token.token_id = initial.value().token_id;
        initial_token.probability = initial.value().probability;
        initial_token.is_stop_token = contains_token(options.stop_tokens, initial_token.token_id);
        const uint64_t initial_sequence_length = sequence_length_;
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
                auto sampled = sample_unlocked(decoded.value().logits, options.sampling);
                if (!sampled)
                    return sampled.error();
                StreamToken token;
                token.index = static_cast<uint32_t>(result.tokens.size());
                token.token_id = sampled.value().token_id;
                token.probability = sampled.value().probability;
                token.is_stop_token = contains_token(options.stop_tokens, token.token_id);
                const uint64_t expected_sequence_length = sequence_length_;
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
            begin_latent_cache_transaction(state_->speculative_layers);
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
            auto discarded_draft_cache = finish_latent_cache_transaction(
                state_->speculative_layers,
                0);
            if (!proposed)
                return proposed.error();
            if (!discarded_draft_cache)
                return discarded_draft_cache.error();
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
            if (options.speculative_confidence_threshold > 0.0f)
            {
                draft_count = std::min(
                    draft_count,
                    proposed.value().confidence_logits.size());
                for (size_t index = 0; index < draft_count; ++index)
                {
                    const float confidence = 1.0f
                                             / (1.0f
                                                + std::exp(
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
            begin_latent_cache_transaction(state_->layers);
            const auto verify_started = std::chrono::steady_clock::now();
            auto target_logits = executor_->execute(
                *model_->compiled_,
                verify_input_ids,
                statistics_scratch_,
                *state_,
                sequence_length_);
            if (!target_logits)
            {
                (void)finish_latent_cache_transaction(
                    state_->layers,
                    0);
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
                        (void)finish_latent_cache_transaction(
                            state_->layers,
                            0);
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
                    auto target_distribution = sampling_distribution(
                        target_logits.value()[accepted],
                        options.sampling);
                    auto draft_distribution = sampling_distribution(
                        proposed.value().logits[accepted],
                        options.sampling);
                    if (!target_distribution || !draft_distribution)
                    {
                        (void)finish_latent_cache_transaction(
                            state_->layers,
                            0);
                        return !target_distribution
                                   ? target_distribution.error()
                                   : draft_distribution.error();
                    }
                    const int32_t token_id = proposed.value().token_ids[accepted];
                    const float target_probability = candidate_probability(
                        target_distribution.value(),
                        token_id);
                    const float draft_probability = candidate_probability(
                        draft_distribution.value(),
                        token_id);
                    if (draft_probability <= 0.0f)
                    {
                        (void)finish_latent_cache_transaction(
                            state_->layers,
                            0);
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
                        rejected_target_distribution = std::move(target_distribution).value();
                        rejected_draft_distribution = std::move(draft_distribution).value();
                        break;
                    }
                    accepted_probabilities.push_back(
                        target_probability);
                }
                ++accepted;
            }
            if (accepted == 0)
            {
                auto rolled_back = finish_latent_cache_transaction(
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
                        random_generator_);
                }
                return sample_unlocked(
                    target_logits.value()[accepted],
                    options.sampling);
            }();
            if (!next)
            {
                (void)finish_latent_cache_transaction(
                    state_->layers,
                    0);
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
                lock.unlock();
                if (decode_text)
                    token.text = decode_text(token.token_id);
                const bool callback_continue = !on_token || on_token(token);
                lock.lock();
                if (sequence_length_ != expected_sequence_length)
                {
                    (void)finish_latent_cache_transaction(
                        state_->layers,
                        0);
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

            auto committed = finish_latent_cache_transaction(
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
