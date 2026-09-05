#include "ncnn/moe/session.h"

#include "executor.h"
#include "sessionstate.h"
#include "expertbackend.h"
#include "graph/compiledmodel.h"
#include "kernels/fastmath.h"
#include "kernels/latentattention.h"
#include "kernels/statecache.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

[[nodiscard]] static Result<void> validate_sampling_options(const SamplingOptions& opt);

[[nodiscard]] static Result<void> sampling_distribution_into(
    std::span<const float> logits,
    const SamplingOptions& opt,
    std::vector<SampledToken>& candidates,
    std::vector<int32_t>& token_ids);

[[nodiscard]] static float candidate_probability(
    std::span<const SampledToken> candidates,
    int32_t token_id) noexcept;

[[nodiscard]] static Result<SampledToken> sample_residual_distribution(
    std::span<const SampledToken> target,
    std::span<const SampledToken> draft,
    size_t vocabulary_size,
    std::vector<float>& residual,
    std::mt19937_64& random_generator);

uint32_t Session::get_max_context_length(const MoeModelDescriptor& descriptor) noexcept
{
    if (descriptor.layers.empty())
        return 0;
    const AttentionDescriptor& attention = descriptor.layers.front().attention;
    return attention.kind == AttentionKind::None ? 0 : attention.max_context_length;
}

Session::Session(ModelPtr _model, const SessionOptions& opt)
    : model(std::move(_model)),
      state(new CpuSessionState),
      random_generator(opt.sampling_seed),
      prefill_chunk_size(opt.prefill_chunk_size),
      use_speculative_context(
          opt.use_speculative_context)
{
    state->use_speculative_context = use_speculative_context;
    stats.expert_token_counts.resize(model->descriptor().expert_count, 0);
}

Session::~Session() = default;

void Session::commit_execution(uint64_t prefill_tokens, uint64_t decode_tokens)
{
    std::swap(stats, stats_scratch);
    stats.prefill_tokens += prefill_tokens;
    stats.decode_tokens += decode_tokens;
    token_count += prefill_tokens + decode_tokens;
    stats.kv_cache_logical_size = state->kv_cache_logical_size();
    stats.kv_cache_allocated_size = state->kv_cache_allocated_size();
}

Result<PrefillResult> Session::prefill_unlocked(std::span<const int32_t> input_ids)
{
    if (input_ids.empty())
        return Error{ErrorCode::InvalidArgument, "prefill requires at least one token"};
    if (input_ids.size() > std::numeric_limits<uint32_t>::max())
        return Error{ErrorCode::InvalidArgument, "prefill token count exceeds uint32 range"};
    const uint32_t max_context_length = get_max_context_length(model->descriptor());
    if (max_context_length > 0 && input_ids.size() > max_context_length - std::min<uint64_t>(token_count, max_context_length))
        return Error{ErrorCode::InvalidArgument, "prefill exceeds the model context length"};

    const CompiledModel& compiled = model_compiled(*model);
    stats_scratch = stats;
    SessionStatistics& updated_statistics = stats_scratch;
    std::vector<float> final_logits;
    size_t processed_tokens = 0;
    while (processed_tokens < input_ids.size())
    {
        const size_t remaining_tokens = input_ids.size() - processed_tokens;
        const size_t chunk_size = prefill_chunk_size == 0 ? remaining_tokens : std::min<size_t>(remaining_tokens, prefill_chunk_size);
        const std::span<const int32_t> chunk = input_ids.subspan(processed_tokens, chunk_size);
        auto chunk_logits = forward_model(compiled, chunk, updated_statistics, *state, token_count + processed_tokens);
        if (!chunk_logits)
            return chunk_logits.error();
        auto speculative_context = update_speculative_context(
            compiled,
            updated_statistics,
            *state);
        if (!speculative_context)
            return speculative_context.error();
        final_logits = std::move(chunk_logits.value().back());
        processed_tokens += chunk_size;
    }

    commit_execution(input_ids.size(), 0);

    PrefillResult result;
    result.logits = std::move(final_logits);
    result.processed_tokens = static_cast<uint32_t>(input_ids.size());
    return result;
}

Result<PrefillResult> Session::prefill(std::span<const int32_t> input_ids)
{
    const std::lock_guard<std::mutex> lock(mutex);
    auto result = prefill_unlocked(input_ids);
    if (result)
        generation_start_stats = stats;
    return result;
}

Result<DecodeResult> Session::decode_unlocked(int32_t input_id)
{
    const uint32_t max_context_length = get_max_context_length(model->descriptor());
    if (max_context_length > 0 && token_count >= max_context_length)
        return Error{ErrorCode::InvalidArgument, "decode exceeds the model context length"};
    const CompiledModel& compiled = model_compiled(*model);
    stats_scratch = stats;
    SessionStatistics& updated_statistics = stats_scratch;
    const std::span<const int32_t> input(&input_id, 1);
    auto all_logits = forward_model(compiled, input, updated_statistics, *state, token_count);
    if (!all_logits)
        return all_logits.error();
    auto speculative_context = update_speculative_context(
        compiled,
        updated_statistics,
        *state);
    if (!speculative_context)
        return speculative_context.error();

    commit_execution(0, 1);

    DecodeResult result;
    result.logits = std::move(all_logits.value().front());
    result.sequence_length = token_count;
    return result;
}

Result<DecodeResult> Session::decode(int32_t input_id)
{
    const std::lock_guard<std::mutex> lock(mutex);
    auto result = decode_unlocked(input_id);
    if (result)
        generation_start_stats = stats;
    return result;
}

Result<void> Session::reset()
{
    const std::lock_guard<std::mutex> lock(mutex);
    token_count = 0;
    stats = {};
    stats.expert_token_counts.resize(model->descriptor().expert_count, 0);
    stats_scratch = {};
    stats_scratch.expert_token_counts.resize(model->descriptor().expert_count, 0);
    state.reset(new CpuSessionState);
    state->use_speculative_context = use_speculative_context;
    generation_start_stats = {};
    generation_active = false;
    generation_input_tokens = 0;
    generation_output_tokens = 0;
    generation_elapsed_microseconds = 0;
    generation_start_time = {};
    generation_first_token_time = {};
    return {};
}

static bool contains_token(const std::vector<int32_t>& tokens, int32_t token)
{
    return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

Result<GenerationResult> Session::generate(std::span<const int32_t> input_ids, const GenerationOptions& opt, TokenStreamCallback on_token, TokenTextDecoder decode_text)
{
    std::unique_lock<std::mutex> lock(mutex);
    if (opt.max_new_tokens == 0)
        return Error{ErrorCode::InvalidArgument, "max_new_tokens must be non-zero"};
    if (!std::isfinite(opt.speculative_confidence_threshold)
        || opt.speculative_confidence_threshold < 0.0f
        || opt.speculative_confidence_threshold > 1.0f)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "speculative_confidence_threshold must be in the range [0, 1]"};
    }
    auto valid_sampling = validate_sampling_options(opt.sampling);
    if (!valid_sampling)
        return valid_sampling.error();

    const CompiledModel& compiled = model_compiled(*model);
    // Keep device-weight admission out of the foreground generation path.
    const ScopedExpertBackendForeground expert_backend_foreground(
        compiled.expert_backend);

    try
    {
        begin_generation(input_ids.size());
        auto ret = generate_unlocked(input_ids, opt, on_token, decode_text, lock);
        finish_generation();
        return ret;
    }
    catch (...)
    {
        if (!lock.owns_lock())
            lock.lock();
        finish_generation();
        throw;
    }
}

Result<bool> Session::emit_token(StreamToken& token, const TokenStreamCallback& on_token, const TokenTextDecoder& decode_text, std::unique_lock<std::mutex>& lock)
{
    const uint64_t expected_token_count = token_count;
    if (generation_output_tokens == 0)
        generation_first_token_time = std::chrono::steady_clock::now();
    ++generation_output_tokens;

    bool continue_generation = true;
    lock.unlock();
    try
    {
        if (decode_text)
            token.text = decode_text(token.token_id);
        continue_generation = !on_token || on_token(token);
    }
    catch (...)
    {
        lock.lock();
        throw;
    }
    lock.lock();
    if (token_count != expected_token_count)
        return Error{ErrorCode::InvalidArgument, "generation callback modified the Session"};
    return continue_generation;
}

Result<GenerationResult> Session::generate_unlocked(std::span<const int32_t> input_ids, const GenerationOptions& opt, const TokenStreamCallback& on_token, const TokenTextDecoder& decode_text, std::unique_lock<std::mutex>& lock)
{
    const CompiledModel& compiled = model_compiled(*model);
    const bool use_speculative = use_speculative_context
                                 && opt.use_speculative
                                 && compiled.speculative.enabled();
    if (use_speculative
        && !state->use_speculative_context
        && token_count != 0)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "cannot enable speculative context after non-speculative decoding"};
    }
    state->use_speculative_context = use_speculative;
    auto prefill_result = prefill_unlocked(input_ids);
    if (!prefill_result)
        return prefill_result.error();

    std::vector<float> logits = std::move(prefill_result).value().logits;
    if (use_speculative)
        return generate_speculative(std::move(logits), opt, on_token, decode_text, lock);

    GenerationResult result;
    result.tokens.reserve(opt.max_new_tokens);
    for (uint32_t index = 0; index < opt.max_new_tokens; ++index)
    {
        auto sampled = sample_unlocked(logits, opt.sampling);
        if (!sampled)
            return sampled.error();

        StreamToken token;
        token.index = index;
        token.token_id = sampled.value().token_id;
        token.probability = sampled.value().probability;
        token.is_stop_token = contains_token(opt.stop_tokens, token.token_id);

        auto callback = emit_token(token, on_token, decode_text, lock);
        if (!callback)
            return callback.error();
        result.tokens.push_back(std::move(token));
        if (!callback.value())
        {
            result.stopped_by_callback = true;
            break;
        }
        if (result.tokens.back().is_stop_token)
        {
            result.stopped_by_stop_token = true;
            break;
        }
        if (index + 1 == opt.max_new_tokens)
            break;

        auto decoded = decode_unlocked(token.token_id);
        if (!decoded)
            return decoded.error();
        logits = std::move(decoded).value().logits;
    }
    return result;
}

Result<GenerationResult> Session::generate_speculative(std::vector<float> logits, const GenerationOptions& opt, const TokenStreamCallback& on_token, const TokenTextDecoder& decode_text, std::unique_lock<std::mutex>& lock)
{
    const CompiledModel& compiled = model_compiled(*model);
    GenerationResult result;
    result.tokens.reserve(opt.max_new_tokens);
    const bool state_cache_transactions = compiled.speculative.kind
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
    try
    {
        auto initial = sample_unlocked(logits, opt.sampling);
        if (!initial)
            return initial.error();
        StreamToken initial_token;
        initial_token.index = 0;
        initial_token.token_id = initial.value().token_id;
        initial_token.probability = initial.value().probability;
        initial_token.is_stop_token = contains_token(opt.stop_tokens, initial_token.token_id);
        auto callback = emit_token(initial_token, on_token, decode_text, lock);
        if (!callback)
            return callback.error();
        result.tokens.push_back(std::move(initial_token));
        if (!callback.value())
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
        while (result.tokens.size() < opt.max_new_tokens)
        {
            const int32_t anchor = result.tokens.back().token_id;
            const size_t remaining = opt.max_new_tokens - result.tokens.size();
            if (remaining == 1 || !speculative_active)
            {
                auto decoded = decode_unlocked(anchor);
                if (!decoded)
                    return decoded.error();
                auto sampled = sample_unlocked(
                    decoded.value().logits,
                    opt.sampling);
                if (!sampled)
                    return sampled.error();
                StreamToken token;
                token.index = static_cast<uint32_t>(result.tokens.size());
                token.token_id = sampled.value().token_id;
                token.probability = sampled.value().probability;
                token.is_stop_token = contains_token(opt.stop_tokens, token.token_id);
                auto callback = emit_token(token, on_token, decode_text, lock);
                if (!callback)
                    return callback.error();
                result.tokens.push_back(std::move(token));
                if (!callback.value())
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
                state->speculative_layers,
                compiled.speculative.block_size);
            if (!draft_transaction)
                return draft_transaction.error();
            auto proposed = propose_speculative(
                compiled,
                anchor,
                stats,
                *state,
                token_count,
                [this, &opt](
                    const std::vector<float>& draft_logits)
                    -> Result<int32_t> {
                    auto sampled = sample_unlocked(
                        draft_logits,
                        opt.sampling);
                    if (!sampled)
                        return sampled.error();
                    return sampled.value().token_id;
                });
            auto discarded_draft_cache = finish_cache_transaction(
                state->speculative_layers,
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
            if (opt.speculative_max_draft_tokens != 0)
            {
                draft_count = std::min(
                    draft_count,
                    static_cast<size_t>(
                        opt.speculative_max_draft_tokens));
            }
            if (opt.speculative_confidence_threshold > 0.0f
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
                        < opt.speculative_confidence_threshold)
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

            stats_scratch = stats;
            auto target_transaction = begin_cache_transaction(
                state->layers,
                verify_input_ids.size());
            if (!target_transaction)
                return target_transaction.error();
            const auto verify_started = std::chrono::steady_clock::now();
            auto execute_target_verify =
                [&]() -> Result<std::vector<std::vector<float>>> {
                if (!state_cache_transactions)
                {
                    return forward_model(
                        compiled,
                        verify_input_ids,
                        stats_scratch,
                        *state,
                        token_count);
                }

                std::vector<std::vector<float>> logits;
                logits.reserve(verify_input_ids.size());
                CpuBatch verified_hidden(
                    verify_input_ids.size(),
                    compiled.descriptor.hidden_size);
                for (size_t index = 0;
                     index < verify_input_ids.size();
                     ++index)
                {
                    const std::span<const int32_t> input(
                        &verify_input_ids[index],
                        1);
                    auto row_logits = forward_model(
                        compiled,
                        input,
                        stats_scratch,
                        *state,
                        token_count + index);
                    if (!row_logits)
                        return row_logits.error();
                    std::vector<std::vector<float>> rows = std::move(row_logits).value();
                    if (rows.size() != 1
                        || state->speculative_main_hidden.rows() != 1)
                    {
                        return Error{
                            ErrorCode::InternalError,
                            "sequential MTP verification produced invalid rows"};
                    }
                    logits.push_back(std::move(rows.front()));
                    std::copy_n(
                        state->speculative_main_hidden.row(0),
                        compiled.descriptor.hidden_size,
                        verified_hidden.row(index));
                }
                state->speculative_main_hidden = std::move(verified_hidden);
                state->speculative_main_hidden_position = token_count;
                return logits;
            };
            auto target_logits = execute_target_verify();
            if (!target_logits)
            {
                auto rolled_back = finish_cache_transaction(
                    state->layers,
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
            std::vector<SampledToken> rejected_target_distribution;
            std::vector<SampledToken> rejected_draft_distribution;
            while (accepted < draft_count)
            {
                if (opt.sampling.temperature == 0.0f)
                {
                    auto target_token = sample_unlocked(
                        target_logits.value()[accepted],
                        opt.sampling);
                    if (!target_token)
                    {
                        auto rolled_back = finish_cache_transaction(
                            state->layers,
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
                    auto target_distribution = sampling_distribution_into(
                        target_logits.value()[accepted],
                        opt.sampling,
                        sampling_candidates[0],
                        sampling_token_ids);
                    auto draft_distribution = sampling_distribution_into(
                        proposed.value().logits[accepted],
                        opt.sampling,
                        sampling_candidates[1],
                        sampling_token_ids);
                    if (!target_distribution || !draft_distribution)
                    {
                        auto rolled_back = finish_cache_transaction(
                            state->layers,
                            0);
                        if (!rolled_back)
                            return rolled_back.error();
                        return !target_distribution
                                   ? target_distribution.error()
                                   : draft_distribution.error();
                    }
                    const int32_t token_id = proposed.value().token_ids[accepted];
                    const float target_probability = candidate_probability(
                        sampling_candidates[0],
                        token_id);
                    const float draft_probability = candidate_probability(
                        sampling_candidates[1],
                        token_id);
                    if (draft_probability <= 0.0f)
                    {
                        auto rolled_back = finish_cache_transaction(
                            state->layers,
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
                    if (distribution(random_generator)
                        >= acceptance_probability)
                    {
                        rejected_target_distribution.assign(
                            sampling_candidates[0].begin(),
                            sampling_candidates[0].end());
                        rejected_draft_distribution.assign(
                            sampling_candidates[1].begin(),
                            sampling_candidates[1].end());
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
                    state->layers,
                    0);
                if (!rolled_back)
                    return rolled_back.error();
                stats_scratch.speculative_verify_time_microseconds += target_verify_microseconds;
                commit_execution(0, 0);
                speculative_active = false;
                continue;
            }
            auto next = [&]() -> Result<SampledToken> {
                if (accepted < draft_count
                    && opt.sampling.temperature > 0.0f)
                {
                    return sample_residual_distribution(
                        rejected_target_distribution,
                        rejected_draft_distribution,
                        target_logits.value()[accepted].size(),
                        sampling_residual,
                        random_generator);
                }
                return sample_unlocked(
                    target_logits.value()[accepted],
                    opt.sampling);
            }();
            if (!next)
            {
                auto rolled_back = finish_cache_transaction(
                    state->layers,
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
                token.is_stop_token = contains_token(opt.stop_tokens, token.token_id);
                auto callback = emit_token(token, on_token, decode_text, lock);
                if (!callback)
                {
                    auto rolled_back = finish_cache_transaction(state->layers, 0);
                    if (!rolled_back)
                        return rolled_back.error();
                    return callback.error();
                }
                result.tokens.push_back(std::move(token));
                ++emitted;
                if (!callback.value())
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
                if (result.tokens.size() == opt.max_new_tokens)
                {
                    continue_generation = false;
                    break;
                }
            }

            auto committed = finish_cache_transaction(
                state->layers,
                emitted);
            if (!committed)
            {
                return committed.error();
            }
            state->speculative_main_hidden.reset(
                emitted,
                state->speculative_main_hidden.columns(),
                false);
            if (state_cache_transactions)
            {
                state->speculative_direct_alignment_ids.resize(
                    emitted);
                for (size_t index = 0; index < emitted; ++index)
                {
                    state->speculative_direct_alignment_ids[index] = output_tokens[index].token_id;
                }
            }
            auto speculative_context = update_speculative_context(
                compiled,
                stats_scratch,
                *state);
            if (!speculative_context)
                return speculative_context.error();
            stats_scratch.speculative_accepted_tokens += emitted > 0 ? emitted - 1 : 0;
            const auto verify_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - verify_started);
            stats_scratch.speculative_verify_time_microseconds += static_cast<uint64_t>(verify_elapsed.count());
            commit_execution(0, emitted);
            if (!continue_generation)
                break;
        }
        return result;
    }
    catch (...)
    {
        // A thrown callback or allocation must not leave either cache transaction active.
        auto target = finish_cache_transaction(state->layers, 0);
        auto draft = finish_cache_transaction(state->speculative_layers, 0);
        if (!target)
            return target.error();
        if (!draft)
            return draft.error();
        throw;
    }
}

static Result<void> validate_sampling_options(const SamplingOptions& opt)
{
    if (!std::isfinite(opt.temperature) || opt.temperature < 0.0f)
        return Error{ErrorCode::InvalidArgument, "temperature must be finite and non-negative"};
    if (!std::isfinite(opt.top_p) || opt.top_p <= 0.0f || opt.top_p > 1.0f)
        return Error{ErrorCode::InvalidArgument, "top_p must be in the range (0, 1]"};
    if (!std::isfinite(opt.min_p) || opt.min_p < 0.0f || opt.min_p > 1.0f)
        return Error{ErrorCode::InvalidArgument, "min_p must be in the range [0, 1]"};
    return {};
}

static Result<void> sampling_distribution_into(
    std::span<const float> logits,
    const SamplingOptions& opt,
    std::vector<SampledToken>& candidates,
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

    const auto logit_order = [logits](int32_t left, int32_t right) {
        const bool left_finite = std::isfinite(logits[left]);
        const bool right_finite = std::isfinite(logits[right]);
        if (left_finite != right_finite)
            return left_finite;
        if (!left_finite)
            return left < right;
        if (logits[left] == logits[right])
            return left < right;
        return logits[left] > logits[right];
    };

    const bool has_top_k = opt.top_k > 0 && opt.top_k < logits.size();
    const bool needs_order = opt.top_p < 1.0f;
    if (has_top_k)
    {
        token_ids.resize(opt.top_k);
        std::iota(token_ids.begin(), token_ids.end(), 0);
        std::make_heap(token_ids.begin(), token_ids.end(), logit_order);
        for (size_t token_id = opt.top_k; token_id < logits.size(); ++token_id)
        {
            const int32_t candidate = static_cast<int32_t>(token_id);
            if (logit_order(candidate, token_ids.front()))
            {
                std::pop_heap(token_ids.begin(), token_ids.end(), logit_order);
                token_ids.back() = candidate;
                std::push_heap(token_ids.begin(), token_ids.end(), logit_order);
            }
        }
        if (needs_order)
            std::sort(token_ids.begin(), token_ids.end(), logit_order);
    }
    else
    {
        token_ids.resize(logits.size());
        std::iota(token_ids.begin(), token_ids.end(), 0);
        if (needs_order)
            std::sort(token_ids.begin(), token_ids.end(), logit_order);
    }

    candidates.reserve(token_ids.size());
    const float maximum = maximum_logit / opt.temperature;
    float normalizer = 0.0f;
    for (int32_t token_id : token_ids)
    {
        const float logit = logits[token_id];
        if (!std::isfinite(logit))
            continue;
        const float probability = float_approximate_exp(logit / opt.temperature - maximum);
        candidates.push_back(SampledToken{token_id, probability});
        normalizer += probability;
    }
    if (candidates.empty() || !std::isfinite(normalizer) || normalizer <= 0.0f)
        return Error{ErrorCode::InvalidArgument, "sampling distribution is invalid"};
    for (SampledToken& candidate : candidates)
        candidate.probability /= normalizer;

    if (opt.min_p > 0.0f)
    {
        float maximum_probability = 0.0f;
        for (const SampledToken& candidate : candidates)
            maximum_probability = std::max(maximum_probability, candidate.probability);
        const float threshold = maximum_probability * opt.min_p;
        size_t count = 0;
        for (const SampledToken& candidate : candidates)
        {
            if (candidate.probability >= threshold)
                candidates[count++] = candidate;
        }
        candidates.resize(count);
    }

    if (opt.top_p < 1.0f)
    {
        float cumulative_probability = 0.0f;
        size_t top_p_count = 0;
        for (const SampledToken& candidate : candidates)
        {
            cumulative_probability += candidate.probability;
            ++top_p_count;
            if (cumulative_probability >= opt.top_p)
                break;
        }
        candidates.resize(top_p_count);
    }

    normalizer = 0.0f;
    for (const SampledToken& candidate : candidates)
        normalizer += candidate.probability;
    if (!std::isfinite(normalizer) || normalizer <= 0.0f)
        return Error{ErrorCode::InvalidArgument, "sampling distribution is invalid"};
    for (SampledToken& candidate : candidates)
        candidate.probability /= normalizer;
    return {};
}

static float candidate_probability(
    std::span<const SampledToken> candidates,
    int32_t token_id) noexcept
{
    for (const SampledToken& candidate : candidates)
    {
        if (candidate.token_id == token_id)
            return candidate.probability;
    }
    return 0.0f;
}

static Result<SampledToken> sample_residual_distribution(
    std::span<const SampledToken> target,
    std::span<const SampledToken> draft,
    size_t vocabulary_size,
    std::vector<float>& residual,
    std::mt19937_64& random_generator)
{
    residual.assign(vocabulary_size, 0.0f);
    for (const SampledToken& candidate : target)
        residual[candidate.token_id] = candidate.probability;
    for (const SampledToken& candidate : draft)
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

Result<SampledToken> Session::sample_unlocked(
    std::span<const float> logits,
    const SamplingOptions& opt)
{
    if (logits.empty())
    {
        return Error{ErrorCode::InvalidArgument, "cannot sample empty logits"};
    }
    auto valid_options = validate_sampling_options(opt);
    if (!valid_options)
        return valid_options.error();
    if (opt.temperature == 0.0f)
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
    auto distribution_result = sampling_distribution_into(
        logits,
        opt,
        sampling_candidates[0],
        sampling_token_ids);
    if (!distribution_result)
        return distribution_result.error();
    const std::vector<SampledToken>& candidates = sampling_candidates[0];

    if (candidates.size() == 1)
        return SampledToken{candidates.front().token_id, 1.0f};

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    const float sample_value = distribution(random_generator);
    float cumulative_probability = 0.0f;
    for (const SampledToken& candidate : candidates)
    {
        cumulative_probability += candidate.probability;
        if (sample_value < cumulative_probability)
            return SampledToken{candidate.token_id, candidate.probability};
    }
    const SampledToken& final_candidate = candidates.back();
    return SampledToken{final_candidate.token_id, final_candidate.probability};
}

Result<SampledToken> Session::sample(std::span<const float> logits, const SamplingOptions& opt)
{
    const std::lock_guard<std::mutex> lock(mutex);
    return sample_unlocked(logits, opt);
}

} // namespace moe
} // namespace ncnn
