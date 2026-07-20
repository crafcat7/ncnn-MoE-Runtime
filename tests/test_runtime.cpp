#include "ncnn/moe/runtime.h"

#include <cmath>
#include <cstdlib>
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

class TemporaryModelPackage
{
public:
    TemporaryModelPackage()
    {
        path_ = std::filesystem::temp_directory_path()
                / ("ncnn_moe_phase0_test_" + std::to_string(std::rand()));
        std::filesystem::create_directories(path_);
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
        path_ = std::filesystem::temp_directory_path()
                / ("ncnn_moe_topk_test_" + std::to_string(std::rand()));
        std::filesystem::create_directories(path_);

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
        path_ = std::filesystem::temp_directory_path()
                / ("ncnn_moe_int8_test_" + std::to_string(std::rand()));
        std::filesystem::create_directories(path_);

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
    AttentionPackage()
    {
        path_ = std::filesystem::temp_directory_path()
                / ("ncnn_moe_attention_test_" + std::to_string(std::rand()));
        std::filesystem::create_directories(path_);

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
  "sliding_window": 2,
  "initial_context_length": 16,
  "max_context_length": 32,
  "rope_theta": 10000.0,
  "rope_scaling_factor": 1.0,
  "norm_epsilon": 0.00001
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
        ncnn::moe::test_topk_selected_weight_normalization_and_combine();
        ncnn::moe::test_int8_expert_linear();
        ncnn::moe::test_invalid_int8_scale_is_rejected();
        ncnn::moe::test_attention_kv_cache_and_reset();
        ncnn::moe::test_loader_reports_adapter_and_weight_errors();
        ncnn::moe::test_phase_zero_rejects_unimplemented_output_mode();
        std::cout << "all ncnn_moe tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
