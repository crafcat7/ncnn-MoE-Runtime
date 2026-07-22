#include "ncnn/moe/runtime.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

void check(bool condition, const char* expression, int line)
{
    if (!condition)
        throw std::runtime_error("check failed at line " + std::to_string(line) + ": " + expression);
}

void check_near(float actual, float expected, float tolerance, int line)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "near check failed at line " + std::to_string(line)
            + ": actual=" + std::to_string(actual)
            + ", expected=" + std::to_string(expected));
    }
}

#define CHECK(expression)                       check(static_cast<bool>(expression), #expression, __LINE__)
#define CHECK_NEAR(actual, expected, tolerance) check_near((actual), (expected), (tolerance), __LINE__)

static std::filesystem::path create_unique_test_directory(const char* prefix)
{
    for (uint32_t attempt = 0; attempt < 1000; ++attempt) {
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const std::filesystem::path candidate = std::filesystem::temp_directory_path()
                                                / (std::string(prefix) + std::to_string(stamp)
                                                   + "_" + std::to_string(attempt));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error))
            return candidate;
        if (error && error != std::make_error_code(std::errc::file_exists))
            throw std::runtime_error("failed to create temporary test directory: " + error.message());
    }
    throw std::runtime_error("failed to allocate a unique temporary test directory");
}

class TemporaryModelPackage
{
public:
    TemporaryModelPackage()
    {
        path_ = create_unique_test_directory("ncnn_moe_phase0_test_");
        write_valid_manifest();
        write_valid_weights();
    }

    ~TemporaryModelPackage()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

    void write_manifest(std::string model_type)
    {
        std::ofstream manifest(path_ / "model.ncnnmoe.json", std::ios::binary | std::ios::trunc);
        manifest << "{\n  \"model_type\": \"" << model_type << "\"\n}\n";
    }

    void truncate_weights()
    {
        std::ofstream weights(path_ / "model.ncnnmoe.bin", std::ios::binary | std::ios::trunc);
        const float one = 1.0f;
        weights.write(reinterpret_cast<const char*>(&one), sizeof(one));
    }

private:
    void write_valid_manifest()
    {
        std::ofstream manifest(path_ / "model.ncnnmoe.json", std::ios::binary);
        manifest << R"({
  "model_type": "tiny_moe",
  "vocabulary_size": 4,
  "hidden_size": 2,
  "intermediate_size": 2,
  "layer_count": 1,
  "expert_count": 2,
  "experts_per_token": 1,
  "expert_activation": "relu",
  "expert_layout": "up_down",
  "normalize_topk_weights": true,
  "use_expert_bias": false,
  "norm_epsilon": 0.00001,
  "weights_file": "model.ncnnmoe.bin"
})";
    }

    void write_valid_weights()
    {
        // Tensor order is the MoeAdapter package contract.
        const std::vector<float> values = {
            // token_embedding.weight [4, 2]
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            -1.0f,
            0.0f,
            // layers.0.pre_ffn_norm.weight [2]
            1.0f,
            1.0f,
            // layers.0.router.weight [2, 2]
            1.0f,
            -1.0f,
            -1.0f,
            1.0f,
            // expert 0 up/down (identity)
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            // expert 1 up (identity), down (2 * identity)
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            2.0f,
            0.0f,
            0.0f,
            2.0f,
            // final_norm.weight [2]
            1.0f,
            1.0f,
            // lm_head.weight [4, 2]
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            -1.0f,
            0.0f,
        };

        std::ofstream weights(path_ / "model.ncnnmoe.bin", std::ios::binary);
        weights.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    std::filesystem::path path_;
};

class WeightedTopKPackage
{
public:
    WeightedTopKPackage()
    {
        path_ = create_unique_test_directory("ncnn_moe_topk_test_");

        std::ofstream manifest(path_ / "model.ncnnmoe.json", std::ios::binary);
        manifest << R"({
  "model_type": "tiny_moe",
  "vocabulary_size": 2,
  "hidden_size": 2,
  "intermediate_size": 2,
  "layer_count": 1,
  "expert_count": 3,
  "experts_per_token": 2,
  "expert_activation": "relu",
  "expert_layout": "up_down",
  "normalize_topk_weights": true,
  "norm_epsilon": 0.00001
})";
        manifest.close();

        const std::vector<float> values = {
            // embedding [2, 2], pre-FFN norm [2]
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            // Router [3, 2]: token 0 ranks expert 0, then 1, then 2.
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            -1.0f,
            0.0f,
            // Expert 0: identity up and down -> x-axis output.
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            // Expert 1: identity up, first intermediate maps to y-axis.
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            // Expert 2: zero projections; it must not be selected.
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            // final norm [2], identity LM head [2, 2]
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
        };
        std::ofstream weights(path_ / "model.ncnnmoe.bin", std::ios::binary);
        weights.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    ~WeightedTopKPackage()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

class Int8ExpertPackage
{
public:
    explicit Int8ExpertPackage(bool invalid_scale = false)
    {
        path_ = create_unique_test_directory("ncnn_moe_int8_test_");

        std::ofstream manifest(path_ / "model.ncnnmoe.json", std::ios::binary);
        manifest << R"({
  "model_type": "tiny_moe",
  "vocabulary_size": 2,
  "hidden_size": 2,
  "intermediate_size": 2,
  "layer_count": 1,
  "expert_count": 1,
  "experts_per_token": 1,
  "expert_activation": "relu",
  "expert_layout": "up_down",
  "expert_weight_dtype": "int8",
  "normalize_topk_weights": true,
  "norm_epsilon": 0.00001
})";
        manifest.close();

        std::ofstream weights(path_ / "model.ncnnmoe.bin", std::ios::binary);
        write_floats(weights, {
                                  // embedding [2, 2], pre-FFN norm [2], router [1, 2]
                                  1.0f,
                                  0.0f,
                                  0.0f,
                                  1.0f,
                                  1.0f,
                                  1.0f,
                                  0.0f,
                                  0.0f,
                              });

        const std::vector<int8_t> identity = {127, 0, 0, 127};
        const float scale = 1.0f / 127.0f;
        write_int8_matrix(weights, identity, {invalid_scale ? 0.0f : scale, scale});
        write_int8_matrix(weights, identity, {scale, scale});

        write_floats(weights, {
                                  // final norm [2], identity LM head [2, 2]
                                  1.0f,
                                  1.0f,
                                  1.0f,
                                  0.0f,
                                  0.0f,
                                  1.0f,
                              });
    }

    ~Int8ExpertPackage()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    static void write_floats(std::ofstream& stream, const std::vector<float>& values)
    {
        stream.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    static void write_int8_matrix(
        std::ofstream& stream,
        const std::vector<int8_t>& values,
        const std::vector<float>& scales)
    {
        stream.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size()));
        write_floats(stream, scales);
    }

    std::filesystem::path path_;
};

class AttentionPackage
{
public:
    explicit AttentionPackage(bool bfloat16_kv_cache = false, uint32_t sliding_window = 2)
    {
        path_ = create_unique_test_directory("ncnn_moe_attention_test_");

        std::ofstream manifest(path_ / "model.ncnnmoe.json", std::ios::binary);
        manifest << R"({
  "model_type": "tiny_moe",
  "vocabulary_size": 2,
  "hidden_size": 2,
  "intermediate_size": 2,
  "layer_count": 1,
  "expert_count": 1,
  "experts_per_token": 1,
  "expert_activation": "relu",
  "expert_layout": "up_down",
  "normalize_topk_weights": true,
  "use_attention": true,
  "attention_head_count": 1,
  "kv_head_count": 1,
  "head_dimension": 2,
  "sliding_window": )"
                 << sliding_window << R"(,
  "initial_context_length": 16,
  "max_context_length": 32,
  "rope_theta": 10000.0,
  "rope_scaling_factor": 1.0,
  "norm_epsilon": 0.00001,
  "kv_cache_dtype": ")"
                 << (bfloat16_kv_cache ? "bfloat16" : "float32") << R"("
})";
        manifest.close();

        const std::vector<float> values = {
            // Embedding and pre-attention norm.
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            // Zero query projection and bias.
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            // Zero key projection and bias.
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            // Identity value projection and zero bias.
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            // Identity output projection, zero bias, and zero sink.
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            // Pre-FFN norm, router, zero up, and zero down.
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            // Final norm and identity LM head.
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
        };
        std::ofstream weights(path_ / "model.ncnnmoe.bin", std::ios::binary);
        weights.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    ~AttentionPackage()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void test_prefill_decode_and_reset()
{
    TemporaryModelPackage package;
    Runtime runtime;
    CHECK(runtime.capabilities().vulkan_attention
          == runtime.capabilities().vulkan_cpu_mix);
    auto model = runtime.load_model(package.path());
    CHECK(model);
    CHECK(model.value()->descriptor().model_type == "tiny_moe");
    CHECK(model.value()->descriptor().layer_count == 1);

    auto session = runtime.create_session(model.value());
    CHECK(session);

    const std::vector<int32_t> prompt = {0, 1, 2};
    auto prefill = session.value()->prefill(prompt);
    CHECK(prefill);
    CHECK(prefill.value().processed_tokens == 3);
    CHECK(prefill.value().logits.values.size() == 4);

    const float normalized_equal = (1.0f + 1.0f / std::sqrt(1.0f + 1e-5f));
    const float final_value = normalized_equal
                              / std::sqrt(normalized_equal * normalized_equal + 1e-5f);
    CHECK_NEAR(prefill.value().logits.values[0], final_value, 1e-5f);
    CHECK_NEAR(prefill.value().logits.values[1], final_value, 1e-5f);
    CHECK_NEAR(prefill.value().logits.values[2], 2.0f * final_value, 1e-5f);
    CHECK_NEAR(prefill.value().logits.values[3], -final_value, 1e-5f);

    CHECK(session.value()->sequence_length() == 3);
    CHECK(session.value()->statistics().prefill_tokens == 3);
    CHECK(session.value()->statistics().decode_tokens == 0);
    CHECK(session.value()->statistics().expert_assignments == 3);
    CHECK(session.value()->statistics().expert_batches == 2);
    CHECK(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({2, 1}));

    auto decode = session.value()->decode(1);
    CHECK(decode);
    CHECK(decode.value().sequence_length == 4);
    CHECK(session.value()->statistics().decode_tokens == 1);
    CHECK(session.value()->statistics().expert_assignments == 4);
    CHECK(session.value()->statistics().expert_batches == 3);
    CHECK(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({2, 2}));

    const float pre_norm = 1.0f / std::sqrt(0.5f + 1e-5f);
    const float expert_one_value = 1.0f + 2.0f * pre_norm;
    const float final_expert_one = expert_one_value
                                   / std::sqrt(expert_one_value * expert_one_value / 2.0f + 1e-5f);
    CHECK_NEAR(decode.value().logits.values[0], 0.0f, 1e-5f);
    CHECK_NEAR(decode.value().logits.values[1], final_expert_one, 1e-5f);
    CHECK_NEAR(decode.value().logits.values[2], final_expert_one, 1e-5f);
    CHECK_NEAR(decode.value().logits.values[3], 0.0f, 1e-5f);

    CHECK(session.value()->reset());
    CHECK(session.value()->sequence_length() == 0);
    CHECK(session.value()->statistics().expert_assignments == 0);
    CHECK(session.value()->statistics().expert_batches == 0);
    CHECK(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({0, 0}));
}

void test_invalid_token_is_transactional()
{
    TemporaryModelPackage package;
    Runtime runtime;
    auto model = runtime.load_model(package.path());
    CHECK(model);
    auto session = runtime.create_session(model.value());
    CHECK(session);

    auto invalid = session.value()->decode(4);
    CHECK(!invalid);
    CHECK(invalid.error().code == ErrorCode::InvalidArgument);
    CHECK(session.value()->sequence_length() == 0);
    CHECK(session.value()->statistics().decode_tokens == 0);
    CHECK(session.value()->statistics().expert_assignments == 0);

    const std::vector<int32_t> empty;
    auto empty_prefill = session.value()->prefill(empty);
    CHECK(!empty_prefill);
    CHECK(empty_prefill.error().code == ErrorCode::InvalidArgument);
}

void test_chunked_prefill_matches_single_batch()
{
    TemporaryModelPackage package;
    Runtime runtime;
    auto model = runtime.load_model(package.path());
    CHECK(model);

    SessionOptions single_batch_options;
    single_batch_options.prefill_chunk_size = 0;
    auto single_batch_session = runtime.create_session(model.value(), single_batch_options);
    CHECK(single_batch_session);

    SessionOptions chunked_options;
    chunked_options.prefill_chunk_size = 2;
    auto chunked_session = runtime.create_session(model.value(), chunked_options);
    CHECK(chunked_session);

    const std::vector<int32_t> prompt = {0, 1, 2};
    auto single_batch = single_batch_session.value()->prefill(prompt);
    auto chunked = chunked_session.value()->prefill(prompt);
    CHECK(single_batch);
    CHECK(chunked);
    CHECK(chunked.value().processed_tokens == prompt.size());
    CHECK(chunked_session.value()->sequence_length() == prompt.size());
    CHECK(chunked_session.value()->statistics().prefill_tokens == prompt.size());
    CHECK(chunked.value().logits.values.size() == single_batch.value().logits.values.size());
    for (size_t index = 0; index < single_batch.value().logits.values.size(); ++index) {
        CHECK_NEAR(
            chunked.value().logits.values[index],
            single_batch.value().logits.values[index],
            1e-5f);
    }
}

void test_topk_selected_weight_normalization_and_combine()
{
    WeightedTopKPackage package;
    Runtime runtime;
    auto model = runtime.load_model(package.path());
    CHECK(model);
    auto session = runtime.create_session(model.value());
    CHECK(session);

    auto result = session.value()->decode(0);
    CHECK(result);

    const float normalized_input = 1.0f / std::sqrt(0.5f + 1e-5f);
    // Expert 2 participates in the global softmax but is removed by Top-2.
    // Renormalization over experts 0 and 1 therefore cancels its probability.
    const float expert_zero_weight = std::exp(normalized_input)
                                     / (std::exp(normalized_input) + 1.0f);
    const float expert_one_weight = 1.0f - expert_zero_weight;
    const float hidden_x = 1.0f + expert_zero_weight * normalized_input;
    const float hidden_y = expert_one_weight * normalized_input;
    const float final_scale = std::sqrt(
        (hidden_x * hidden_x + hidden_y * hidden_y) / 2.0f + 1e-5f);

    CHECK_NEAR(result.value().logits.values[0], hidden_x / final_scale, 1e-5f);
    CHECK_NEAR(result.value().logits.values[1], hidden_y / final_scale, 1e-5f);
    CHECK(session.value()->statistics().expert_assignments == 2);
    CHECK(session.value()->statistics().expert_batches == 2);
    CHECK(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({1, 1, 0}));
}

void test_int8_expert_linear()
{
    Int8ExpertPackage package;
    Runtime runtime;
    auto model = runtime.load_model(package.path());
    CHECK(model);
    CHECK(model.value()->descriptor().layers[0].ffn.moe.expert_weight_dtype == DType::Int8);

    auto session = runtime.create_session(model.value());
    CHECK(session);
    auto result = session.value()->decode(0);
    CHECK(result);

    const float normalized_input = 1.0f / std::sqrt(0.5f + 1e-5f);
    const float hidden = 1.0f + normalized_input;
    const float expected = hidden / std::sqrt(hidden * hidden / 2.0f + 1e-5f);
    CHECK_NEAR(result.value().logits.values[0], expected, 1e-5f);
    CHECK_NEAR(result.value().logits.values[1], 0.0f, 1e-5f);
    CHECK(session.value()->statistics().expert_assignments == 1);
    CHECK(session.value()->statistics().expert_batches == 1);
}

void test_invalid_int8_scale_is_rejected()
{
    Int8ExpertPackage package(true);
    Runtime runtime;
    auto model = runtime.load_model(package.path());
    CHECK(!model);
    CHECK(model.error().code == ErrorCode::InvalidModel);
}

void test_attention_kv_cache_and_reset()
{
    AttentionPackage package;
    Runtime runtime;
    auto model = runtime.load_model(package.path());
    CHECK(model);
    CHECK(model.value()->descriptor().layers[0].use_attention);

    auto session = runtime.create_session(model.value());
    CHECK(session);
    const std::vector<int32_t> prompt = {0};
    auto prefill = session.value()->prefill(prompt);
    CHECK(prefill);
    auto cached_decode = session.value()->decode(1);
    CHECK(cached_decode);
    CHECK(cached_decode.value().logits.values[0] > 0.1f);

    CHECK(session.value()->reset());
    auto uncached_decode = session.value()->decode(1);
    CHECK(uncached_decode);
    CHECK_NEAR(uncached_decode.value().logits.values[0], 0.0f, 1e-6f);
}

void test_bfloat16_ring_kv_cache()
{
    AttentionPackage float32_package;
    AttentionPackage bfloat16_package(true);
    Runtime runtime;
    RuntimeOptions cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto float32_model = runtime.load_model(float32_package.path(), cpu_options);
    auto bfloat16_model = runtime.load_model(bfloat16_package.path(), cpu_options);
    CHECK(float32_model);
    CHECK(bfloat16_model);
    CHECK(bfloat16_model.value()->descriptor().kv_cache_dtype == DType::BFloat16);

    auto float32_session = runtime.create_session(float32_model.value());
    auto bfloat16_session = runtime.create_session(bfloat16_model.value());
    CHECK(float32_session);
    CHECK(bfloat16_session);

    const std::vector<int32_t> prompt = {0, 1, 0, 1, 0, 1, 0, 1};
    CHECK(float32_session.value()->prefill(prompt));
    CHECK(bfloat16_session.value()->prefill(prompt));
    CHECK(bfloat16_session.value()->statistics().kv_cache_logical_bytes
          == float32_session.value()->statistics().kv_cache_logical_bytes / 2);
    CHECK(bfloat16_session.value()->statistics().kv_cache_allocated_bytes
          == float32_session.value()->statistics().kv_cache_allocated_bytes / 2);

    for (uint32_t index = 0; index < 16; ++index)
        CHECK(bfloat16_session.value()->decode(static_cast<int32_t>(index % 2)));
    CHECK(bfloat16_session.value()->statistics().kv_cache_allocated_bytes
          <= bfloat16_session.value()->statistics().kv_cache_logical_bytes * 16);
}

void test_sampling_and_streaming_generation()
{
    TemporaryModelPackage package;
    Runtime runtime;
    auto model = runtime.load_model(package.path());
    CHECK(model);

    SessionOptions session_options;
    session_options.sampling_seed = 1234;
    auto session = runtime.create_session(model.value(), session_options);
    CHECK(session);

    LogitsOutput logits;
    logits.values = {1.0f, 2.0f, 3.0f};
    SamplingOptions greedy_options;
    greedy_options.temperature = 0.0f;
    auto greedy = session.value()->sample(logits, greedy_options);
    CHECK(greedy);
    CHECK(greedy.value().token_id == 2);
    CHECK_NEAR(greedy.value().probability, 1.0f, 1e-6f);

    SamplingOptions top_k_options;
    top_k_options.top_k = 1;
    auto top_k = session.value()->sample(logits, top_k_options);
    CHECK(top_k);
    CHECK(top_k.value().token_id == 2);

    SamplingOptions invalid_options;
    invalid_options.top_p = 0.0f;
    auto invalid = session.value()->sample(logits, invalid_options);
    CHECK(!invalid);
    CHECK(invalid.error().code == ErrorCode::InvalidArgument);

    GenerationOptions generation_options;
    generation_options.max_new_tokens = 3;
    generation_options.sampling.temperature = 0.0f;
    std::vector<StreamToken> streamed;
    auto generated = session.value()->generate(
        std::vector<int32_t>{0},
        generation_options,
        [&streamed](const StreamToken& token) {
            streamed.push_back(token);
            return true;
        },
        [](int32_t token_id) { return "<" + std::to_string(token_id) + ">"; });
    CHECK(generated);
    CHECK(generated.value().tokens.size() == 3);
    CHECK(streamed.size() == 3);
    CHECK(generated.value().tokens[0].text
          == "<" + std::to_string(generated.value().tokens[0].token_id) + ">");
    CHECK(session.value()->sequence_length() == 3);

    auto stopped_session = runtime.create_session(model.value(), session_options);
    CHECK(stopped_session);
    auto stopped = stopped_session.value()->generate(
        std::vector<int32_t>{0},
        generation_options,
        [](const StreamToken&) { return false; });
    CHECK(stopped);
    CHECK(stopped.value().stopped_by_callback);
    CHECK(stopped.value().tokens.size() == 1);
    CHECK(stopped_session.value()->sequence_length() == 1);
}

void test_loader_reports_adapter_and_weight_errors()
{
    {
        TemporaryModelPackage package;
        package.write_manifest("unknown_family");
        Runtime runtime;
        auto model = runtime.load_model(package.path());
        CHECK(!model);
        CHECK(model.error().code == ErrorCode::UnsupportedModel);
    }

    {
        TemporaryModelPackage package;
        package.truncate_weights();
        Runtime runtime;
        auto model = runtime.load_model(package.path() / "model.ncnnmoe.json");
        CHECK(!model);
        CHECK(model.error().code == ErrorCode::InvalidModel);
    }
}

void test_backend_capabilities_and_hybrid_execution()
{
    TemporaryModelPackage package;
    Runtime runtime;
    CHECK(runtime.capabilities().cpu_execution);
    CHECK(runtime.capabilities().vulkan_cpu_prefetch == runtime.capabilities().vulkan_cpu_mix);

    RuntimeOptions automatic_options;
    automatic_options.hybrid_mode = HybridMode::Auto;
    auto automatic_model = runtime.load_model(package.path(), automatic_options);
    CHECK(automatic_model);
    CHECK(automatic_model.value()->hybrid_mode()
          == (runtime.capabilities().vulkan_cpu_mix ? HybridMode::HybridExperts : HybridMode::CpuOnly));
    auto automatic_session = runtime.create_session(automatic_model.value());
    CHECK(automatic_session);
    const std::vector<int32_t> packed_prompt = {0, 1, 2, 3};
    auto automatic_prefill = automatic_session.value()->prefill(packed_prompt);
    CHECK(automatic_prefill);
    if (runtime.capabilities().vulkan_cpu_mix)
        CHECK(automatic_session.value()->statistics().vulkan_linear_dispatches > 0);
    else
        CHECK(automatic_session.value()->statistics().vulkan_linear_dispatches == 0);

    RuntimeOptions cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto cpu_model = runtime.load_model(package.path(), cpu_options);
    CHECK(cpu_model);
    auto cpu_session = runtime.create_session(cpu_model.value());
    CHECK(cpu_session);
    auto cpu_prefill = cpu_session.value()->prefill(packed_prompt);
    CHECK(cpu_prefill);
    CHECK(cpu_prefill.value().logits.values.size() == automatic_prefill.value().logits.values.size());
    for (size_t index = 0; index < cpu_prefill.value().logits.values.size(); ++index) {
        CHECK_NEAR(
            automatic_prefill.value().logits.values[index],
            cpu_prefill.value().logits.values[index],
            1e-4f);
    }

    RuntimeOptions hybrid_options;
    hybrid_options.hybrid_mode = HybridMode::HybridExperts;
    auto hybrid_model = runtime.load_model(package.path(), hybrid_options);
    if (runtime.capabilities().vulkan_cpu_mix) {
        CHECK(hybrid_model);
        CHECK(hybrid_model.value()->hybrid_mode() == HybridMode::HybridExperts);

        AttentionPackage attention_package;
        auto attention_model = runtime.load_model(attention_package.path(), hybrid_options);
        CHECK(attention_model);
        auto attention_session = runtime.create_session(attention_model.value());
        CHECK(attention_session);
        const std::vector<int32_t> attention_prompt = {0, 1, 0, 1};
        auto attention_prefill = attention_session.value()->prefill(attention_prompt);
        CHECK(attention_prefill);
        CHECK(attention_session.value()->statistics().vulkan_linear_dispatches == 4);
        CHECK(attention_session.value()->statistics().vulkan_attention_blocks == 1);
        CHECK(attention_session.value()->statistics().kv_cache_logical_bytes > 0);
        CHECK(attention_session.value()->statistics().kv_cache_allocated_bytes
              >= attention_session.value()->statistics().kv_cache_logical_bytes);

        auto cpu_attention_model = runtime.load_model(attention_package.path(), cpu_options);
        CHECK(cpu_attention_model);
        auto cpu_attention_session = runtime.create_session(cpu_attention_model.value());
        CHECK(cpu_attention_session);
        auto cpu_attention_prefill = cpu_attention_session.value()->prefill(attention_prompt);
        CHECK(cpu_attention_prefill);
        CHECK(cpu_attention_prefill.value().logits.values.size()
              == attention_prefill.value().logits.values.size());
        for (size_t index = 0; index < cpu_attention_prefill.value().logits.values.size(); ++index) {
            CHECK_NEAR(
                attention_prefill.value().logits.values[index],
                cpu_attention_prefill.value().logits.values[index],
                1e-4f);
        }
        auto attention_decode = attention_session.value()->decode(1);
        auto cpu_attention_decode = cpu_attention_session.value()->decode(1);
        CHECK(attention_decode);
        CHECK(cpu_attention_decode);
        CHECK(attention_session.value()->statistics().vulkan_attention_blocks == 2);
        for (size_t index = 0; index < cpu_attention_decode.value().logits.values.size(); ++index) {
            CHECK_NEAR(
                attention_decode.value().logits.values[index],
                cpu_attention_decode.value().logits.values[index],
                1e-4f);
        }

        AttentionPackage full_attention_package(false, 0);
        auto full_attention_model = runtime.load_model(full_attention_package.path(), hybrid_options);
        CHECK(full_attention_model);
        SessionOptions chunked_attention_options;
        chunked_attention_options.prefill_chunk_size = 2;
        auto chunked_attention_session = runtime.create_session(
            full_attention_model.value(), chunked_attention_options);
        CHECK(chunked_attention_session);
        auto chunked_attention_prefill = chunked_attention_session.value()->prefill(attention_prompt);
        CHECK(chunked_attention_prefill);
        CHECK(chunked_attention_session.value()->statistics().vulkan_attention_blocks == 2);
        CHECK(chunked_attention_session.value()->statistics().kv_cache_allocated_bytes
              > chunked_attention_session.value()->statistics().kv_cache_logical_bytes);

        auto full_cpu_model = runtime.load_model(full_attention_package.path(), cpu_options);
        CHECK(full_cpu_model);
        auto full_cpu_session = runtime.create_session(full_cpu_model.value());
        CHECK(full_cpu_session);
        auto full_cpu_prefill = full_cpu_session.value()->prefill(attention_prompt);
        CHECK(full_cpu_prefill);
        for (size_t index = 0; index < full_cpu_prefill.value().logits.values.size(); ++index) {
            CHECK_NEAR(
                chunked_attention_prefill.value().logits.values[index],
                full_cpu_prefill.value().logits.values[index],
                1e-4f);
        }
        auto chunked_attention_decode = chunked_attention_session.value()->decode(1);
        auto full_cpu_decode = full_cpu_session.value()->decode(1);
        CHECK(chunked_attention_decode);
        CHECK(full_cpu_decode);
        for (size_t index = 0; index < full_cpu_decode.value().logits.values.size(); ++index) {
            CHECK_NEAR(
                chunked_attention_decode.value().logits.values[index],
                full_cpu_decode.value().logits.values[index],
                1e-4f);
        }

        RuntimeOptions prefetch_options;
        prefetch_options.hybrid_mode = HybridMode::VulkanWithCpuPrefetch;
        auto prefetch_model = runtime.load_model(package.path(), prefetch_options);
        CHECK(prefetch_model);
        CHECK(prefetch_model.value()->hybrid_mode() == HybridMode::VulkanWithCpuPrefetch);
        auto prefetch_session = runtime.create_session(prefetch_model.value());
        CHECK(prefetch_session);
        CHECK(prefetch_session.value()->prefill(packed_prompt));
        CHECK(prefetch_session.value()->statistics().expert_prefetches > 0);
        CHECK(prefetch_session.value()->statistics().expert_prefetch_bytes > 0);
    }
    else {
        CHECK(!hybrid_model);
        CHECK(hybrid_model.error().code == ErrorCode::UnsupportedModel);

        RuntimeOptions prefetch_options;
        prefetch_options.hybrid_mode = HybridMode::VulkanWithCpuPrefetch;
        auto prefetch_model = runtime.load_model(package.path(), prefetch_options);
        CHECK(!prefetch_model);
        CHECK(prefetch_model.error().code == ErrorCode::UnsupportedModel);
    }
}

void test_phase_zero_rejects_unimplemented_output_mode()
{
    TemporaryModelPackage package;
    Runtime runtime;
    auto model = runtime.load_model(package.path());
    CHECK(model);

    SessionOptions options;
    options.logits_output_mode = LogitsOutputMode::TopKCandidates;
    auto session = runtime.create_session(model.value(), options);
    CHECK(!session);
    CHECK(session.error().code == ErrorCode::UnsupportedModel);
}

} // namespace moe
} // namespace ncnn

int main()
{
    try {
        ncnn::moe::test_prefill_decode_and_reset();
        ncnn::moe::test_invalid_token_is_transactional();
        ncnn::moe::test_chunked_prefill_matches_single_batch();
        ncnn::moe::test_topk_selected_weight_normalization_and_combine();
        ncnn::moe::test_int8_expert_linear();
        ncnn::moe::test_invalid_int8_scale_is_rejected();
        ncnn::moe::test_attention_kv_cache_and_reset();
        ncnn::moe::test_bfloat16_ring_kv_cache();
        ncnn::moe::test_sampling_and_streaming_generation();
        ncnn::moe::test_loader_reports_adapter_and_weight_errors();
        ncnn::moe::test_backend_capabilities_and_hybrid_execution();
        ncnn::moe::test_phase_zero_rejects_unimplemented_output_mode();
        std::cout << "all ncnn_moe tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
