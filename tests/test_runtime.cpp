#include "ncnn/moe/runtime.h"

#include "cpu_mxfp4.h"
#include "cpu_ops.h"
#include "cpu_topology.h"
#include "expert_cache.h"
#include "moe_adapter.h"
#include "ncnn_linear.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
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
    explicit AttentionPackage(
        bool bfloat16_kv_cache = false,
        uint32_t sliding_window = 2,
        bool attention_bias = true,
        bool attention_sinks = true)
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
  "attention_bias": )"
                 << (attention_bias ? "true" : "false") << R"(,
  "attention_sinks": )"
                 << (attention_sinks ? "true" : "false") << R"(,
  "norm_epsilon": 0.00001,
  "kv_cache_dtype": ")"
                 << (bfloat16_kv_cache ? "bfloat16" : "float32") << R"("
})";
        manifest.close();

        std::vector<float> values;
        auto append = [&values](std::initializer_list<float> additions) {
            values.insert(values.end(), additions.begin(), additions.end());
        };
        // Embedding and pre-attention norm.
        append({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f});
        // Zero query and key projections.
        append({0.0f, 0.0f, 0.0f, 0.0f});
        if (attention_bias)
            append({0.0f, 0.0f});
        append({0.0f, 0.0f, 0.0f, 0.0f});
        if (attention_bias)
            append({0.0f, 0.0f});
        // Identity value and output projections.
        append({1.0f, 0.0f, 0.0f, 1.0f});
        if (attention_bias)
            append({0.0f, 0.0f});
        append({1.0f, 0.0f, 0.0f, 1.0f});
        if (attention_bias)
            append({0.0f, 0.0f});
        if (attention_sinks)
            append({0.0f});
        // Pre-FFN norm, router, zero up, and zero down.
        append({
            1.0f, 1.0f,
            0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
        });
        // Final norm and identity LM head.
        append({1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f});
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
    CHECK_NEAR(decode.value().logits.values[1], final_expert_one, 2e-5f);
    CHECK_NEAR(decode.value().logits.values[2], final_expert_one, 2e-5f);
    CHECK_NEAR(decode.value().logits.values[3], 0.0f, 1e-5f);

    CHECK(session.value()->reset());
    CHECK(session.value()->sequence_length() == 0);
    CHECK(session.value()->statistics().expert_assignments == 0);
    CHECK(session.value()->statistics().expert_batches == 0);
    CHECK(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({0, 0}));
}

void test_ncnn_linear_operator()
{
#if NCNN_MOE_WITH_NCNN
    TensorData matrix;
    matrix.dtype = DType::Float32;
    matrix.shape = {4, 3};
    matrix.float32_data = {
        0.1234f, -0.9876f, 1.2345f,
        -0.2222f, 0.3333f, -0.4444f,
        1.1111f, -1.2222f, 0.5555f,
        -0.8765f, 0.7654f, -0.6543f,
    };
    TensorData bias;
    bias.dtype = DType::Float32;
    bias.shape = {4};
    bias.float32_data = {0.1357f, -0.2468f, 0.3579f, -0.4680f};

    const auto linear = NcnnLinearOperator::create(matrix, &bias);
    CHECK(linear);

    CpuBatch input(2, 3);
    input.row(0)[0] = 0.2345f;
    input.row(0)[1] = -1.3456f;
    input.row(0)[2] = 2.4567f;
    input.row(1)[0] = -0.5678f;
    input.row(1)[1] = 1.6789f;
    input.row(1)[2] = -2.7891f;
    CpuBatch output;
    CHECK(linear->forward(input, output));
    CHECK(output.rows() == 2);
    CHECK(output.columns() == 4);
    for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
        for (uint32_t column = 0; column < output.columns(); ++column) {
            float expected = bias.float32_data[column];
            for (uint32_t input_column = 0; input_column < input.columns(); ++input_column) {
                expected += matrix.float32_data[column * input.columns() + input_column]
                            * input.row(row_index)[input_column];
            }
            CHECK_NEAR(output.row(row_index)[column], expected, 1e-5f);
        }
    }

    TensorData bfloat_matrix;
    bfloat_matrix.dtype = DType::BFloat16;
    bfloat_matrix.shape = matrix.shape;
    for (float value : matrix.float32_data)
        bfloat_matrix.bfloat16_data.push_back(float_to_bfloat16(value));
    TensorData bfloat_bias;
    bfloat_bias.dtype = DType::BFloat16;
    bfloat_bias.shape = bias.shape;
    for (float value : bias.float32_data)
        bfloat_bias.bfloat16_data.push_back(float_to_bfloat16(value));

    const auto bfloat_linear = NcnnLinearOperator::create(bfloat_matrix, &bfloat_bias);
    CHECK(bfloat_linear);
    CHECK(bfloat_linear->forward(input, output));
    for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
        for (uint32_t column = 0; column < output.columns(); ++column) {
            float expected = bfloat16_to_float(bfloat_bias.bfloat16_data[column]);
            for (uint32_t input_column = 0; input_column < input.columns(); ++input_column) {
                expected += bfloat16_to_float(
                                bfloat_matrix.bfloat16_data[column * input.columns() + input_column])
                            * input.row(row_index)[input_column];
            }
            CHECK_NEAR(output.row(row_index)[column], expected, 1e-5f);
        }
    }

    if (NcnnLinearOperator::vulkan_device_count() > 0) {
        const auto vulkan_linear = NcnnLinearOperator::create(
            matrix,
            &bias,
            NcnnLinearDevice::Vulkan);
        CHECK(vulkan_linear);
        const NcnnVulkanRuntimeCounters initial_counters
            = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
        for (uint32_t iteration = 0; iteration < 4; ++iteration) {
            CHECK(vulkan_linear->forward(input, output));
            for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
                for (uint32_t column = 0; column < output.columns(); ++column) {
                    float expected = bias.float32_data[column];
                    for (uint32_t input_column = 0; input_column < input.columns(); ++input_column) {
                        expected += matrix.float32_data[column * input.columns() + input_column]
                                    * input.row(row_index)[input_column];
                    }
                    CHECK_NEAR(output.row(row_index)[column], expected, 1e-4f);
                }
            }
        }
        const NcnnVulkanRuntimeCounters final_counters
            = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
        CHECK(final_counters.compute_submissions - initial_counters.compute_submissions == 4);
        CHECK(final_counters.batch_uploads - initial_counters.batch_uploads == 4);
        CHECK(final_counters.batch_downloads - initial_counters.batch_downloads == 4);
        CHECK(final_counters.auxiliary_uploads - initial_counters.auxiliary_uploads == 0);
        CHECK(final_counters.staging_slot_resizes - initial_counters.staging_slot_resizes
                  + final_counters.staging_slot_reuses - initial_counters.staging_slot_reuses
              == 8);
        CHECK(final_counters.staging_slot_reuses - initial_counters.staging_slot_reuses >= 4);
        CHECK(final_counters.staging_slot_acquisitions
                  - initial_counters.staging_slot_acquisitions
              == 4);
    }
#endif
}

void test_mxfp4_cpu_kernel_and_fused_gate_up()
{
    TensorData matrix;
    matrix.dtype = DType::MxFp4;
    matrix.shape = {4, 32};
    matrix.mxfp4_scales = {127, 127, 127, 127};
    matrix.mxfp4_blocks.resize(64);
    for (size_t row = 0; row < 4; ++row) {
        for (size_t byte = 0; byte < 16; ++byte) {
            const uint8_t low = static_cast<uint8_t>((byte + row) % 16);
            const uint8_t high = static_cast<uint8_t>((byte * 3 + row + 1) % 16);
            matrix.mxfp4_blocks[row * 16 + byte]
                = static_cast<uint8_t>(low | (high << 4));
        }
    }
    CpuBatch input(2, 32);
    for (size_t row = 0; row < input.rows(); ++row) {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(
                static_cast<int>(column % 7) - 3)
                                     * (row == 0 ? 0.25f : -0.125f);
    }
    static constexpr float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    auto scalar_row = [&](size_t matrix_row, size_t input_row) {
        float sum = 0.0f;
        for (size_t byte = 0; byte < 16; ++byte) {
            const uint8_t packed = matrix.mxfp4_blocks[matrix_row * 16 + byte];
            sum += values[packed & 0x0f] * input.row(input_row)[byte * 2];
            sum += values[packed >> 4] * input.row(input_row)[byte * 2 + 1];
        }
        return sum;
    };

    const CpuBatch projected = linear_batch(matrix, input);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row) {
        for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row) {
            CHECK_NEAR(
                projected.row(input_row)[matrix_row],
                scalar_row(matrix_row, input_row),
                1e-5f);
        }
    }

    CpuBatch decode_input(1, 32);
    std::copy_n(input.row(0), input.columns(), decode_input.row(0));
    const CpuBatch decoded = linear_batch(matrix, decode_input);
    for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row) {
        CHECK_NEAR(
            decoded.row(0)[matrix_row],
            scalar_row(matrix_row, 0),
            1e-5f);
    }

    TensorData odd_matrix = matrix;
    odd_matrix.shape[0] = 3;
    odd_matrix.mxfp4_blocks.resize(3 * 16);
    odd_matrix.mxfp4_scales.resize(3);
    const CpuBatch odd_projected = linear_batch(odd_matrix, input);
    CHECK(odd_projected.columns() == 3);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row) {
        for (size_t matrix_row = 0; matrix_row < 3; ++matrix_row) {
            CHECK_NEAR(
                odd_projected.row(input_row)[matrix_row],
                scalar_row(matrix_row, input_row),
                1e-5f);
        }
    }

    TensorData bias;
    bias.dtype = DType::Float32;
    bias.shape = {4};
    bias.float32_data = {0.25f, -0.5f, 0.75f, -1.0f};
    const CpuBatch fused = fused_mxfp4_gate_up_batch(matrix, &bias, input, 7.0f);
    CHECK(fused.rows() == input.rows());
    CHECK(fused.columns() == 2);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row) {
        for (size_t column = 0; column < fused.columns(); ++column) {
            const float gate = std::min(
                scalar_row(column * 2, input_row) + bias.float32_data[column * 2],
                7.0f);
            const float linear = std::clamp(
                scalar_row(column * 2 + 1, input_row)
                    + bias.float32_data[column * 2 + 1],
                -7.0f,
                7.0f);
            const float expected
                = gate / (1.0f + std::exp(-1.702f * gate)) * (linear + 1.0f);
            CHECK_NEAR(fused.row(input_row)[column], expected, 1e-5f);
        }
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    CHECK(mxfp4_kernel_kind() == MxFp4KernelKind::ArmNeon);
#endif
    CHECK(std::string(mxfp4_kernel_name()).size() > 0);
}

void test_file_backed_mxfp4_expert_cache()
{
    const std::filesystem::path directory
        = create_unique_test_directory("ncnn_moe_expert_cache_test_");
    const std::filesystem::path blocks_path = directory / "blocks.bin";
    const std::filesystem::path scales_path = directory / "scales.bin";
    {
        std::vector<uint8_t> blocks(64);
        for (size_t index = 0; index < blocks.size(); ++index)
            blocks[index] = static_cast<uint8_t>(index);
        std::ofstream stream(blocks_path, std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(blocks.data()),
            static_cast<std::streamsize>(blocks.size()));
    }
    {
        const std::vector<uint8_t> scales = {101, 102, 103, 104};
        std::ofstream stream(scales_path, std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(scales.data()),
            static_cast<std::streamsize>(scales.size()));
    }

    auto file_backed = [&](uint64_t block_offset, uint64_t scale_offset) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {1, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = blocks_path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_bytes = 16;
        storage->scales_path = scales_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_bytes = 1;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };

    const TensorData gate_zero = file_backed(0, 0);
    const TensorData down_zero = file_backed(16, 1);
    const TensorData gate_one = file_backed(32, 2);
    const TensorData down_one = file_backed(48, 3);
    Mxfp4ExpertCache cache(34);
    {
        auto first = cache.acquire_pair(gate_zero, down_zero);
        CHECK(first);
        CHECK(!first.value().cache_hit);
        CHECK(first.value().bytes_read == 34);
        CHECK(first.value().gate_up->mxfp4_blocks.front() == 0);
        CHECK(first.value().down->mxfp4_blocks.front() == 16);
        CHECK(first.value().gate_up->mxfp4_scales.front() == 101);
        CHECK(first.value().down->mxfp4_scales.front() == 102);
    }
    {
        auto hit = cache.acquire_pair(gate_zero, down_zero);
        CHECK(hit);
        CHECK(hit.value().cache_hit);
        CHECK(hit.value().bytes_read == 0);
    }
    {
        auto second = cache.acquire_pair(gate_one, down_one);
        CHECK(second);
        CHECK(!second.value().cache_hit);
        CHECK(second.value().gate_up->mxfp4_blocks.front() == 32);
        CHECK(second.value().down->mxfp4_scales.front() == 104);
    }
    const ExpertCacheStatistics statistics = cache.statistics();
    CHECK(statistics.hits == 1);
    CHECK(statistics.misses == 2);
    CHECK(statistics.evictions == 1);
    CHECK(statistics.bytes_read == 68);
    CHECK(statistics.resident_bytes == 34);

    Mxfp4ExpertCache concurrent(68, 2);
    CHECK(concurrent.request_pair(gate_zero, down_zero));
    bool first_ok = false;
    bool second_ok = false;
    bool first_hit = false;
    bool second_hit = false;
    std::thread first_waiter([&] {
        auto lease = concurrent.acquire_pair(gate_zero, down_zero);
        first_ok = static_cast<bool>(lease);
        first_hit = lease && lease.value().cache_hit;
    });
    std::thread second_waiter([&] {
        auto lease = concurrent.acquire_pair(gate_zero, down_zero);
        second_ok = static_cast<bool>(lease);
        second_hit = lease && lease.value().cache_hit;
    });
    first_waiter.join();
    second_waiter.join();
    CHECK(first_ok);
    CHECK(second_ok);
    CHECK(first_hit != second_hit);
    CHECK(concurrent.statistics().misses == 1);
    CHECK(concurrent.statistics().hits == 1);
    CHECK(concurrent.statistics().queued_reads == 1);

    Mxfp4ExpertCache speculative(68, 1);
    CHECK(speculative.prefetch_pair(gate_zero, down_zero));
    auto prefetched = speculative.acquire_pair(gate_zero, down_zero);
    CHECK(prefetched);
    CHECK(speculative.statistics().speculative_reads == 1);
    CHECK(speculative.statistics().queued_reads == 1);

    Mxfp4ExpertCache pressure(34, 1);
    auto pinned = pressure.acquire_pair(gate_zero, down_zero);
    CHECK(pinned);
    CHECK(pressure.prefetch_pair(gate_one, down_one));
    CHECK(pressure.statistics().speculative_reads == 0);
    auto exhausted = pressure.acquire_pair(gate_one, down_one);
    CHECK(!exhausted);
    CHECK(exhausted.error().code == ErrorCode::InvalidArgument);

    const TensorData truncated_gate = file_backed(60, 0);
    Mxfp4ExpertCache retryable(34, 1);
    auto failed_read = retryable.acquire_pair(truncated_gate, down_zero);
    CHECK(!failed_read);
    CHECK(failed_read.error().code == ErrorCode::IoError);
    auto retried_read = retryable.acquire_pair(truncated_gate, down_zero);
    CHECK(!retried_read);
    CHECK(retried_read.error().code == ErrorCode::IoError);
    CHECK(retryable.statistics().misses == 2);
    CHECK(retryable.statistics().resident_bytes == 0);

    Mxfp4ExpertCache undersized(33);
    auto rejected = undersized.acquire_pair(gate_zero, down_zero);
    CHECK(!rejected);
    CHECK(rejected.error().code == ErrorCode::InvalidArgument);

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_cpu_topology_parsing_and_partitioning()
{
    CHECK(
        parse_linux_cpu_list("0-3,8,10-11")
        == std::vector<uint32_t>({0, 1, 2, 3, 8, 10, 11}));
    CHECK(
        parse_linux_cpu_list("4,2-4,2")
        == std::vector<uint32_t>({2, 3, 4}));
    CHECK(parse_linux_cpu_list("3-1").empty());
    CHECK(parse_linux_cpu_list("1,,2").empty());
    CHECK(parse_linux_cpu_list("1,").empty());

    CpuTopology flat;
    flat.allowed_cpus = {2, 4, 7, 9, 12};
    const std::vector<std::vector<uint32_t> > flat_partitions
        = partition_cpu_topology(flat, 2);
    CHECK(flat_partitions.size() == 2);
    CHECK(flat_partitions[0] == std::vector<uint32_t>({2, 4, 7}));
    CHECK(flat_partitions[1] == std::vector<uint32_t>({9, 12}));

    CpuTopology numa;
    numa.allowed_cpus = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    numa.numa_nodes = {
        {0, 1, 2, 3},
        {4, 5, 6, 7, 8, 9, 10, 11},
    };
    const std::vector<std::vector<uint32_t> > numa_partitions
        = partition_cpu_topology(numa, 4);
    CHECK(numa_partitions.size() == 4);
    std::vector<uint32_t> partitioned_cpus;
    for (const std::vector<uint32_t>& partition : numa_partitions) {
        CHECK(!partition.empty());
        const bool first_node = partition.front() < 4;
        for (uint32_t cpu : partition)
            CHECK((cpu < 4) == first_node);
        partitioned_cpus.insert(
            partitioned_cpus.end(), partition.begin(), partition.end());
    }
    std::sort(partitioned_cpus.begin(), partitioned_cpus.end());
    CHECK(partitioned_cpus == numa.allowed_cpus);
}

void test_cross_session_batch_scheduler()
{
    TemporaryModelPackage package;
    Runtime runtime;
    auto model = runtime.load_model(package.path());
    CHECK(model);
    auto first = runtime.create_session(model.value());
    auto second = runtime.create_session(model.value());
    CHECK(first);
    CHECK(second);

    SchedulerOptions options;
    options.worker_count = 2;
    auto scheduler = runtime.create_scheduler(options);
    CHECK(scheduler);
    auto future = scheduler.value()->submit_decode({
        {first.value(), 0},
        {second.value(), 1},
    });
    std::vector<Result<DecodeResult> > results = future.get();
    CHECK(results.size() == 2);
    CHECK(results[0]);
    CHECK(results[1]);
    CHECK(first.value()->sequence_length() == 1);
    CHECK(second.value()->sequence_length() == 1);

    auto ordered_session = runtime.create_session(model.value());
    auto reference_session = runtime.create_session(model.value());
    CHECK(ordered_session);
    CHECK(reference_session);
    auto ordered_first = scheduler.value()->submit_decode({
        {ordered_session.value(), 0},
    });
    auto ordered_second = scheduler.value()->submit_decode({
        {ordered_session.value(), 1},
    });
    auto reference_first = reference_session.value()->decode(0);
    auto reference_second = reference_session.value()->decode(1);
    CHECK(reference_first);
    CHECK(reference_second);
    std::vector<Result<DecodeResult> > ordered_first_result = ordered_first.get();
    std::vector<Result<DecodeResult> > ordered_second_result = ordered_second.get();
    CHECK(ordered_first_result[0]);
    CHECK(ordered_second_result[0]);
    CHECK(ordered_first_result[0].value().sequence_length == 1);
    CHECK(ordered_second_result[0].value().sequence_length == 2);
    for (size_t index = 0;
         index < reference_second.value().logits.values.size();
         ++index) {
        CHECK_NEAR(
            ordered_second_result[0].value().logits.values[index],
            reference_second.value().logits.values[index],
            1e-5f);
    }

    auto duplicate_future = scheduler.value()->submit_decode({
        {first.value(), 0},
        {first.value(), 1},
    });
    std::vector<Result<DecodeResult> > duplicate_results = duplicate_future.get();
    CHECK(!duplicate_results[0]);
    CHECK(!duplicate_results[1]);
    CHECK(first.value()->sequence_length() == 1);
    const SchedulerStatistics statistics = scheduler.value()->statistics();
    CHECK(statistics.worker_count == 2);
    CHECK(statistics.expert_threads_per_worker >= 1);
    CHECK(statistics.submitted_batches == 4);
    CHECK(statistics.submitted_requests == 6);
    CHECK(statistics.completed_requests == 6);
    CHECK(statistics.rejected_requests == 2);
    CHECK(statistics.max_batch_size == 2);
    CHECK(statistics.max_in_flight >= 2);

    SchedulerOptions mismatched_affinity;
    mismatched_affinity.worker_count = 2;
    mismatched_affinity.worker_cpu_sets = {{0}};
    auto mismatched_scheduler = runtime.create_scheduler(mismatched_affinity);
    CHECK(!mismatched_scheduler);
    CHECK(mismatched_scheduler.error().code == ErrorCode::InvalidArgument);

    SchedulerOptions empty_affinity;
    empty_affinity.worker_cpu_sets = {{}};
    auto empty_affinity_scheduler = runtime.create_scheduler(empty_affinity);
    CHECK(!empty_affinity_scheduler);
    CHECK(empty_affinity_scheduler.error().code == ErrorCode::InvalidArgument);
#if !defined(__linux__)
    SchedulerOptions unsupported_affinity;
    unsupported_affinity.worker_cpu_sets = {{0}};
    auto unsupported_scheduler = runtime.create_scheduler(unsupported_affinity);
    CHECK(!unsupported_scheduler);
    CHECK(unsupported_scheduler.error().code == ErrorCode::UnsupportedModel);
#endif

    if (runtime.capabilities().vulkan_cpu_mix) {
        AttentionPackage attention_package;
        RuntimeOptions hybrid_options;
        hybrid_options.hybrid_mode = HybridMode::HybridExperts;
        RuntimeOptions cpu_options;
        cpu_options.hybrid_mode = HybridMode::CpuOnly;
        auto hybrid_model = runtime.load_model(
            attention_package.path(), hybrid_options);
        auto cpu_model = runtime.load_model(attention_package.path(), cpu_options);
        CHECK(hybrid_model);
        CHECK(cpu_model);

        SchedulerOptions pipeline_options;
        pipeline_options.worker_count = 4;
        auto pipeline_scheduler = runtime.create_scheduler(pipeline_options);
        CHECK(pipeline_scheduler);
        std::vector<SessionPtr> hybrid_sessions;
        std::vector<SessionPtr> cpu_sessions;
        for (uint32_t index = 0; index < 4; ++index) {
            auto hybrid_session = runtime.create_session(hybrid_model.value());
            auto cpu_session = runtime.create_session(cpu_model.value());
            CHECK(hybrid_session);
            CHECK(cpu_session);
            hybrid_sessions.push_back(hybrid_session.value());
            cpu_sessions.push_back(cpu_session.value());
        }

        constexpr uint32_t pipeline_rounds = 6;
        for (uint32_t round = 0; round < pipeline_rounds; ++round) {
            std::vector<DecodeBatchRequest> requests;
            for (uint32_t session_index = 0;
                 session_index < hybrid_sessions.size();
                 ++session_index) {
                requests.push_back({
                    hybrid_sessions[session_index],
                    static_cast<int32_t>((round + session_index) % 2),
                });
            }
            std::vector<Result<DecodeResult> > pipeline_results
                = pipeline_scheduler.value()->submit_decode(std::move(requests)).get();
            CHECK(pipeline_results.size() == hybrid_sessions.size());
            for (uint32_t session_index = 0;
                 session_index < hybrid_sessions.size();
                 ++session_index) {
                auto cpu_result = cpu_sessions[session_index]->decode(
                    static_cast<int32_t>((round + session_index) % 2));
                CHECK(pipeline_results[session_index]);
                CHECK(cpu_result);
                CHECK(pipeline_results[session_index].value().sequence_length
                      == round + 1);
                CHECK(
                    pipeline_results[session_index].value().logits.values.size()
                    == cpu_result.value().logits.values.size());
                for (size_t logit = 0;
                     logit < cpu_result.value().logits.values.size();
                     ++logit) {
                    CHECK_NEAR(
                        pipeline_results[session_index]
                            .value()
                            .logits.values[logit],
                        cpu_result.value().logits.values[logit],
                        1e-4f);
                }
            }
        }
        for (const SessionPtr& session : hybrid_sessions) {
            const SessionStatistics session_statistics = session->statistics();
            CHECK(session_statistics.vulkan_attention_blocks == pipeline_rounds);
            CHECK(
                session_statistics.vulkan_compute_submissions
                == pipeline_rounds * 2);
            CHECK(
                session_statistics.vulkan_staging_slot_acquisitions
                == pipeline_rounds * 2);
        }
        const SchedulerStatistics pipeline_statistics
            = pipeline_scheduler.value()->statistics();
        CHECK(pipeline_statistics.completed_requests == pipeline_rounds * 4);
        CHECK(pipeline_statistics.max_in_flight >= 4);
    }
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
    if (runtime.capabilities().openmp_expert_parallelism)
        CHECK(session.value()->statistics().expert_parallel_tasks == 2);
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

void test_attention_graph_without_bias_or_sink()
{
    AttentionPackage package(false, 0, false, false);
    std::ifstream manifest_stream(package.path() / "model.ncnnmoe.json");
    const std::string manifest_json{
        std::istreambuf_iterator<char>(manifest_stream),
        std::istreambuf_iterator<char>()};
    ModelPackage model_package;
    model_package.root = package.path();
    model_package.manifest.model_type = "tiny_moe";
    model_package.manifest.raw_json = manifest_json;
    MoeAdapter adapter;
    auto descriptor = adapter.parse_model(model_package);
    CHECK(descriptor);
    auto mapping = adapter.map_weights(model_package, descriptor.value());
    CHECK(mapping);
    descriptor.value().layers[0].use_attention = false;
    descriptor.value().layers[0].use_moe = false;
    ModelCompiler compiler;
    auto graph_driven_model = compiler.compile(
        std::move(descriptor).value(),
        std::move(mapping).value(),
        HybridMode::CpuOnly);
    CHECK(graph_driven_model);
    CHECK(graph_driven_model.value().layers[0].use_attention);

    Runtime runtime;
    RuntimeOptions cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto cpu_model = runtime.load_model(package.path(), cpu_options);
    CHECK(cpu_model);
    const LayerDescriptor& layer = cpu_model.value()->descriptor().layers[0];
    const std::vector<ModelNodeType> expected_nodes = {
        ModelNodeType::RmsNorm,
        ModelNodeType::FusedQkv,
        ModelNodeType::Rope,
        ModelNodeType::Sdpa,
        ModelNodeType::Projection,
        ModelNodeType::RmsNorm,
        ModelNodeType::Router,
        ModelNodeType::TopK,
        ModelNodeType::ExpertGroup,
        ModelNodeType::Combine,
    };
    CHECK(layer.nodes.size() == expected_nodes.size());
    for (size_t index = 0; index < expected_nodes.size(); ++index)
        CHECK(layer.nodes[index].type == expected_nodes[index]);
    CHECK(!layer.attention.use_bias);
    CHECK(!layer.attention.use_sinks);
    const CompiledLayerPlan& cpu_plan
        = cpu_model.value()->execution_plan()[0];
    CHECK(cpu_plan.nodes.size() == expected_nodes.size());
    for (const CompiledNodePlan& node : cpu_plan.nodes)
        CHECK(node.backend == ExecutionBackend::Cpu);

    auto cpu_session = runtime.create_session(cpu_model.value());
    CHECK(cpu_session);
    const std::vector<int32_t> prompt = {0, 1, 0};
    auto cpu_prefill = cpu_session.value()->prefill(prompt);
    CHECK(cpu_prefill);

    if (runtime.capabilities().vulkan_attention) {
        RuntimeOptions hybrid_options;
        hybrid_options.hybrid_mode = HybridMode::HybridExperts;
        auto hybrid_model = runtime.load_model(package.path(), hybrid_options);
        CHECK(hybrid_model);
        const CompiledLayerPlan& hybrid_plan
            = hybrid_model.value()->execution_plan()[0];
        CHECK(hybrid_plan.nodes.size() == expected_nodes.size());
        for (size_t node_index = 0;
             node_index < hybrid_plan.nodes.size();
             ++node_index) {
            CHECK(
                hybrid_plan.nodes[node_index].backend
                == (node_index < 5
                        ? ExecutionBackend::Vulkan
                        : ExecutionBackend::Cpu));
        }
        auto hybrid_session = runtime.create_session(hybrid_model.value());
        CHECK(hybrid_session);
        auto hybrid_prefill = hybrid_session.value()->prefill(prompt);
        CHECK(hybrid_prefill);
        CHECK(hybrid_session.value()->statistics().vulkan_attention_blocks == 1);
        CHECK(hybrid_prefill.value().logits.values.size()
              == cpu_prefill.value().logits.values.size());
        for (size_t index = 0;
             index < cpu_prefill.value().logits.values.size();
             ++index) {
            CHECK_NEAR(
                hybrid_prefill.value().logits.values[index],
                cpu_prefill.value().logits.values[index],
                1e-4f);
        }
    }
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
    if (runtime.capabilities().vulkan_cpu_mix) {
        const SessionStatistics& statistics = automatic_session.value()->statistics();
        CHECK(statistics.vulkan_linear_dispatches == 1);
        CHECK(statistics.vulkan_compute_submissions == 1);
        CHECK(statistics.vulkan_batch_uploads == 1);
        CHECK(statistics.vulkan_batch_downloads == 1);
        CHECK(statistics.vulkan_auxiliary_uploads == 0);
        CHECK(statistics.vulkan_auxiliary_upload_bytes == 0);
        CHECK(statistics.vulkan_staging_slot_resizes
                  + statistics.vulkan_staging_slot_reuses
              == 2);
        CHECK(statistics.vulkan_staging_slot_acquisitions == 1);
    }
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
        CHECK(attention_session.value()->statistics().vulkan_attention_blocks == 1);
        CHECK(attention_session.value()->statistics().vulkan_linear_dispatches == 3);
        CHECK(attention_session.value()->statistics().vulkan_compute_submissions == 2);
        CHECK(attention_session.value()->statistics().vulkan_batch_uploads == 2);
        CHECK(attention_session.value()->statistics().vulkan_batch_downloads == 2);
        CHECK(attention_session.value()->statistics().vulkan_auxiliary_uploads == 3);
        CHECK(attention_session.value()->statistics().vulkan_auxiliary_upload_bytes > 0);
        CHECK(attention_session.value()->statistics().vulkan_staging_slot_resizes
                  + attention_session.value()->statistics().vulkan_staging_slot_reuses
              == 7);
        CHECK(attention_session.value()->statistics().vulkan_staging_slot_reuses > 0);
        CHECK(attention_session.value()->statistics().vulkan_staging_slot_acquisitions == 2);
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
        ncnn::moe::test_ncnn_linear_operator();
        ncnn::moe::test_mxfp4_cpu_kernel_and_fused_gate_up();
        ncnn::moe::test_file_backed_mxfp4_expert_cache();
        ncnn::moe::test_cpu_topology_parsing_and_partitioning();
        ncnn::moe::test_cross_session_batch_scheduler();
        ncnn::moe::test_prefill_decode_and_reset();
        ncnn::moe::test_invalid_token_is_transactional();
        ncnn::moe::test_chunked_prefill_matches_single_batch();
        ncnn::moe::test_topk_selected_weight_normalization_and_combine();
        ncnn::moe::test_int8_expert_linear();
        ncnn::moe::test_invalid_int8_scale_is_rejected();
        ncnn::moe::test_attention_kv_cache_and_reset();
        ncnn::moe::test_bfloat16_ring_kv_cache();
        ncnn::moe::test_attention_graph_without_bias_or_sink();
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
