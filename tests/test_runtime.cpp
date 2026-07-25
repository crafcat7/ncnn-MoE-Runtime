#include "ncnn/moe/runtime.h"

#include "kernels/cpu_mxfp4.h"
#include "kernels/cpu_ops.h"
#include "engine/cpu_topology.h"
#include "storage/expert_cache.h"
#include "fixture_model_adapter.h"
#include "storage/mapped_file.h"
#include "graph/memory_planner.h"
#include "backends/ncnn/ncnn_linear.h"
#include "ncnn/moe/expert_dispatcher.h"
#include "models/safetensors.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

static void check(bool condition)
{
    if (!condition)
        throw std::runtime_error("test check failed");
}

static void check_near(float actual, float expected, float tolerance)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "near check failed: actual=" + std::to_string(actual)
            + ", expected=" + std::to_string(expected));
    }
}

class TestRuntime final : public Runtime
{
public:
    TestRuntime()
    {
        register_adapter(std::make_shared<FixtureModelAdapter>());
    }
};

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
        std::ofstream manifest(path_ / "config.json", std::ios::binary | std::ios::trunc);
        manifest << "{\n  \"model_type\": \"" << model_type << "\"\n}\n";
    }

    void truncate_weights()
    {
        std::ofstream weights(path_ / "model.test.bin", std::ios::binary | std::ios::trunc);
        const float one = 1.0f;
        weights.write(reinterpret_cast<const char*>(&one), sizeof(one));
    }

private:
    void write_valid_manifest()
    {
        std::ofstream manifest(path_ / "config.json", std::ios::binary);
        manifest << R"({
  "model_type": "test_moe",
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
  "weights_file": "model.test.bin"
})";
    }

    void write_valid_weights()
    {
        const std::vector<float> values = {
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            -1.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            -1.0f,
            -1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            2.0f,
            0.0f,
            0.0f,
            2.0f,
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            -1.0f,
            0.0f,
        };

        std::ofstream weights(path_ / "model.test.bin", std::ios::binary);
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

        std::ofstream manifest(path_ / "config.json", std::ios::binary);
        manifest << R"({
  "model_type": "test_moe",
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
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            -1.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
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
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
        };
        std::ofstream weights(path_ / "model.test.bin", std::ios::binary);
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

        std::ofstream manifest(path_ / "config.json", std::ios::binary);
        manifest << R"({
  "model_type": "test_moe",
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

        std::ofstream weights(path_ / "model.test.bin", std::ios::binary);
        write_floats(weights, {
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

        std::ofstream manifest(path_ / "config.json", std::ios::binary);
        manifest << R"({
  "model_type": "test_moe",
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
        append({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f});
        append({0.0f, 0.0f, 0.0f, 0.0f});
        if (attention_bias)
            append({0.0f, 0.0f});
        append({0.0f, 0.0f, 0.0f, 0.0f});
        if (attention_bias)
            append({0.0f, 0.0f});
        append({1.0f, 0.0f, 0.0f, 1.0f});
        if (attention_bias)
            append({0.0f, 0.0f});
        append({1.0f, 0.0f, 0.0f, 1.0f});
        if (attention_bias)
            append({0.0f, 0.0f});
        if (attention_sinks)
            append({0.0f});
        append({
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
        });
        append({1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f});
        std::ofstream weights(path_ / "model.test.bin", std::ios::binary);
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
    TestRuntime runtime;
    check(static_cast<bool>(has_flag(
                                runtime.capabilities().flags,
                                RuntimeCapabilityVulkanAttention)
                            == has_flag(
                                runtime.capabilities().flags,
                                RuntimeCapabilityVulkanCpuMix)));
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    check(static_cast<bool>(model.value()->descriptor().model_type == "test_moe"));
    check(static_cast<bool>(model.value()->descriptor().layer_count == 1));

    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));

    const std::vector<int32_t> prompt = {0, 1, 2};
    auto prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill));
    check(static_cast<bool>(prefill.value().processed_tokens == 3));
    check(static_cast<bool>(prefill.value().logits.values.size() == 4));

    const float normalized_equal = (1.0f + 1.0f / std::sqrt(1.0f + 1e-5f));
    const float final_value = normalized_equal
                              / std::sqrt(normalized_equal * normalized_equal + 1e-5f);
    check_near(prefill.value().logits.values[0], final_value, 1e-5f);
    check_near(prefill.value().logits.values[1], final_value, 1e-5f);
    check_near(prefill.value().logits.values[2], 2.0f * final_value, 1e-5f);
    check_near(prefill.value().logits.values[3], -final_value, 1e-5f);

    check(static_cast<bool>(session.value()->sequence_length() == 3));
    check(static_cast<bool>(session.value()->statistics().prefill_tokens == 3));
    check(static_cast<bool>(session.value()->statistics().decode_tokens == 0));
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 3));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 2));
    check(static_cast<bool>(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({2, 1})));

    auto decode = session.value()->decode(1);
    check(static_cast<bool>(decode));
    check(static_cast<bool>(decode.value().sequence_length == 4));
    check(static_cast<bool>(session.value()->statistics().decode_tokens == 1));
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 4));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 3));
    check(static_cast<bool>(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({2, 2})));

    const float pre_norm = 1.0f / std::sqrt(0.5f + 1e-5f);
    const float expert_one_value = 1.0f + 2.0f * pre_norm;
    const float final_expert_one = expert_one_value
                                   / std::sqrt(expert_one_value * expert_one_value / 2.0f + 1e-5f);
    check_near(decode.value().logits.values[0], 0.0f, 1e-5f);
    check_near(decode.value().logits.values[1], final_expert_one, 2e-5f);
    check_near(decode.value().logits.values[2], final_expert_one, 2e-5f);
    check_near(decode.value().logits.values[3], 0.0f, 1e-5f);

    check(static_cast<bool>(session.value()->reset()));
    check(static_cast<bool>(session.value()->sequence_length() == 0));
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 0));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 0));
    check(static_cast<bool>(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({0, 0})));
}

void test_ncnn_linear_operator()
{
#if NCNN_MOE_WITH_NCNN
    TensorData matrix;
    matrix.dtype = DType::Float32;
    matrix.shape = {4, 3};
    matrix.float32_data = {
        0.1234f,
        -0.9876f,
        1.2345f,
        -0.2222f,
        0.3333f,
        -0.4444f,
        1.1111f,
        -1.2222f,
        0.5555f,
        -0.8765f,
        0.7654f,
        -0.6543f,
    };
    TensorData bias;
    bias.dtype = DType::Float32;
    bias.shape = {4};
    bias.float32_data = {0.1357f, -0.2468f, 0.3579f, -0.4680f};

    const auto linear = NcnnLinearOperator::create(matrix, &bias);
    check(static_cast<bool>(linear));

    CpuBatch input(2, 3);
    input.row(0)[0] = 0.2345f;
    input.row(0)[1] = -1.3456f;
    input.row(0)[2] = 2.4567f;
    input.row(1)[0] = -0.5678f;
    input.row(1)[1] = 1.6789f;
    input.row(1)[2] = -2.7891f;
    CpuBatch output;
    check(static_cast<bool>(linear->forward(input, output)));
    check(static_cast<bool>(output.rows() == 2));
    check(static_cast<bool>(output.columns() == 4));
    for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
        for (uint32_t column = 0; column < output.columns(); ++column) {
            float expected = bias.float32_data[column];
            for (uint32_t input_column = 0; input_column < input.columns(); ++input_column) {
                expected += matrix.float32_data[column * input.columns() + input_column]
                            * input.row(row_index)[input_column];
            }
            check_near(output.row(row_index)[column], expected, 1e-5f);
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
    check(static_cast<bool>(bfloat_linear));
    check(static_cast<bool>(bfloat_linear->forward(input, output)));
    for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
        for (uint32_t column = 0; column < output.columns(); ++column) {
            float expected = bfloat16_to_float(bfloat_bias.bfloat16_data[column]);
            for (uint32_t input_column = 0; input_column < input.columns(); ++input_column) {
                expected += bfloat16_to_float(
                                bfloat_matrix.bfloat16_data[column * input.columns() + input_column])
                            * input.row(row_index)[input_column];
            }
            check_near(output.row(row_index)[column], expected, 1e-5f);
        }
    }

    if (NcnnLinearOperator::vulkan_device_count() > 0) {
        const auto vulkan_linear = NcnnLinearOperator::create(
            matrix,
            &bias,
            NcnnLinearDevice::Vulkan);
        check(static_cast<bool>(vulkan_linear));
        const NcnnVulkanRuntimeCounters initial_counters
            = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
        for (uint32_t iteration = 0; iteration < 4; ++iteration) {
            check(static_cast<bool>(vulkan_linear->forward(input, output)));
            for (size_t row_index = 0; row_index < input.rows(); ++row_index) {
                for (uint32_t column = 0; column < output.columns(); ++column) {
                    float expected = bias.float32_data[column];
                    for (uint32_t input_column = 0; input_column < input.columns(); ++input_column) {
                        expected += matrix.float32_data[column * input.columns() + input_column]
                                    * input.row(row_index)[input_column];
                    }
                    check_near(output.row(row_index)[column], expected, 1e-4f);
                }
            }
        }
        const NcnnVulkanRuntimeCounters final_counters
            = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
        check(static_cast<bool>(final_counters.compute_submissions - initial_counters.compute_submissions == 4));
        check(static_cast<bool>(final_counters.batch_uploads - initial_counters.batch_uploads == 4));
        check(static_cast<bool>(final_counters.batch_downloads - initial_counters.batch_downloads == 4));
        check(static_cast<bool>(final_counters.auxiliary_uploads - initial_counters.auxiliary_uploads == 0));
        check(static_cast<bool>(final_counters.staging_slot_resizes - initial_counters.staging_slot_resizes
                                    + final_counters.staging_slot_reuses - initial_counters.staging_slot_reuses
                                == 8));
        check(static_cast<bool>(final_counters.staging_slot_reuses - initial_counters.staging_slot_reuses >= 4));
        check(static_cast<bool>(final_counters.staging_slot_acquisitions
                                    - initial_counters.staging_slot_acquisitions
                                == 4));
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
            check_near(
                projected.row(input_row)[matrix_row],
                scalar_row(matrix_row, input_row),
                1e-5f);
        }
    }

    CpuBatch decode_input(1, 32);
    std::copy_n(input.row(0), input.columns(), decode_input.row(0));
    const CpuBatch decoded = linear_batch(matrix, decode_input);
    for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row) {
        check_near(
            decoded.row(0)[matrix_row],
            scalar_row(matrix_row, 0),
            1e-5f);
    }

    TensorData odd_matrix = matrix;
    odd_matrix.shape[0] = 3;
    odd_matrix.mxfp4_blocks.resize(3 * 16);
    odd_matrix.mxfp4_scales.resize(3);
    const CpuBatch odd_projected = linear_batch(odd_matrix, input);
    check(static_cast<bool>(odd_projected.columns() == 3));
    for (size_t input_row = 0; input_row < input.rows(); ++input_row) {
        for (size_t matrix_row = 0; matrix_row < 3; ++matrix_row) {
            check_near(
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
    check(static_cast<bool>(fused.rows() == input.rows()));
    check(static_cast<bool>(fused.columns() == 2));
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
            check_near(fused.row(input_row)[column], expected, 1e-5f);
        }
    }

    constexpr uint32_t test_block_count = 4;
    constexpr size_t test_token_count = 3;
    constexpr size_t test_input_columns = test_block_count * 32;
    constexpr size_t test_input_stride = test_input_columns + 5;
    constexpr size_t test_output_stride = 3;
    std::vector<uint8_t> first_packed(test_block_count * 16);
    std::vector<uint8_t> second_packed(test_block_count * 16);
    for (size_t index = 0; index < first_packed.size(); ++index) {
        first_packed[index] = static_cast<uint8_t>(
            ((index * 5 + 3) % 16) | (((index * 7 + 1) % 16) << 4));
        second_packed[index] = static_cast<uint8_t>(
            ((index * 11 + 2) % 16) | (((index * 3 + 9) % 16) << 4));
    }
    const std::array<uint8_t, test_block_count> first_scales
        = {125, 127, 128, 126};
    const std::array<uint8_t, test_block_count> second_scales
        = {128, 124, 127, 129};
    std::vector<float> strided_input(test_token_count * test_input_stride);
    for (size_t token = 0; token < test_token_count; ++token) {
        for (size_t column = 0; column < test_input_stride; ++column) {
            strided_input[token * test_input_stride + column]
                = static_cast<float>(
                      static_cast<int>((column * 13 + token * 5) % 23) - 11)
                  * 0.03125f;
        }
    }
    auto reference_dot = [&](
                             const std::vector<uint8_t>& packed,
                             const std::array<uint8_t, test_block_count>& scales,
                             size_t token) {
        float sum = 0.0f;
        for (uint32_t block = 0; block < test_block_count; ++block) {
            float block_sum = 0.0f;
            for (uint32_t byte = 0; byte < 16; ++byte) {
                const uint8_t value = packed[block * 16 + byte];
                const size_t input_offset
                    = token * test_input_stride + block * 32 + byte * 2;
                block_sum
                    += values[value & 0x0f] * strided_input[input_offset];
                block_sum
                    += values[value >> 4] * strided_input[input_offset + 1];
            }
            sum += block_sum
                   * std::ldexp(
                       1.0f,
                       static_cast<int>(scales[block]) - 127);
        }
        return sum;
    };

    check_near(
        mxfp4_dot(
            first_packed.data(),
            first_scales.data(),
            test_block_count,
            strided_input.data()),
        reference_dot(first_packed, first_scales, 0),
        1e-4f);

    std::vector<float> gemm_output(
        test_token_count * test_output_stride,
        -999.0f);
    mxfp4_gemm_row(
        first_packed.data(),
        first_scales.data(),
        test_block_count,
        strided_input.data(),
        test_input_stride,
        test_token_count,
        gemm_output.data(),
        test_output_stride);
    for (size_t token = 0; token < test_token_count; ++token) {
        check_near(
            gemm_output[token * test_output_stride],
            reference_dot(first_packed, first_scales, token),
            1e-4f);
    }

    std::vector<float> paired_first(
        test_token_count * test_output_stride,
        -999.0f);
    std::vector<float> paired_second(
        test_token_count * test_output_stride,
        -999.0f);
    mxfp4_matmul_rows2(
        first_packed.data(),
        first_scales.data(),
        second_packed.data(),
        second_scales.data(),
        test_block_count,
        strided_input.data(),
        test_input_stride,
        test_token_count,
        paired_first.data(),
        test_output_stride,
        paired_second.data(),
        test_output_stride);
    for (size_t token = 0; token < test_token_count; ++token) {
        check_near(
            paired_first[token * test_output_stride],
            reference_dot(first_packed, first_scales, token),
            1e-4f);
        check_near(
            paired_second[token * test_output_stride],
            reference_dot(second_packed, second_scales, token),
            1e-4f);
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    check(static_cast<bool>(mxfp4_kernel_kind() == MxFp4KernelKind::ArmNeon));
#endif
    check(static_cast<bool>(std::string(mxfp4_kernel_name()).size() > 0));
}

class TestExpertVictimCache final : public IExpertVictimCache
{
public:
    explicit TestExpertVictimCache(uint64_t capacity_bytes)
        : capacity_bytes_(capacity_bytes)
    {
    }

    void admit(
        std::string key,
        std::shared_ptr<const TensorData> gate_up,
        std::shared_ptr<const TensorData> down) override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        ExpertVictimPair pair;
        pair.gate_up = std::make_shared<TensorData>(*gate_up);
        pair.down = std::make_shared<TensorData>(*down);
        entries_[std::move(key)] = std::move(pair);
        ++statistics_.admissions;
        ++statistics_.stores;
        statistics_.resident_bytes
            += gate_up->mxfp4_blocks.size()
               + gate_up->mxfp4_scales.size()
               + down->mxfp4_blocks.size()
               + down->mxfp4_scales.size();
    }

    std::optional<ExpertVictimPair> restore(
        const std::string& key,
        const TensorData&,
        const TensorData&) override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        const auto existing = entries_.find(key);
        if (existing == entries_.end()) {
            ++statistics_.misses;
            return std::nullopt;
        }
        ++statistics_.hits;
        return existing->second;
    }

    ExpertVictimCacheStatistics statistics() const override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return statistics_;
    }

    uint64_t capacity_bytes() const noexcept override
    {
        return capacity_bytes_;
    }

private:
    uint64_t capacity_bytes_ = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ExpertVictimPair> entries_;
    ExpertVictimCacheStatistics statistics_;
};

void test_mapped_file_range_and_shared_buffer()
{
    const std::filesystem::path directory
        = create_unique_test_directory("ncnn_moe_mapped_file_test_");
    const std::filesystem::path path = directory / "range.bin";
    const size_t file_size = 3 * 4096 + 257;
    std::vector<uint8_t> expected(file_size);
    for (size_t index = 0; index < expected.size(); ++index)
        expected[index] = static_cast<uint8_t>((index * 29 + 7) % 251);
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(expected.data()),
            static_cast<std::streamsize>(expected.size()));
    }

    {
        constexpr size_t offset = 123;
        constexpr size_t byte_count = 2 * 4096 + 91;
        auto mapped = MappedFileRange::open(
            path,
            offset,
            byte_count);
        check(static_cast<bool>(mapped));
        mapped.value()->prefault();
        check(static_cast<bool>(mapped.value()->size() == byte_count));
        check(static_cast<bool>(mapped.value()->data()[0] == expected[offset]));
        check(static_cast<bool>(
            mapped.value()->data()[byte_count - 1]
            == expected[offset + byte_count - 1]));

        MxFp4ByteBuffer shared = mapped.value()->share_bytes();
        check(static_cast<bool>(shared.size() == byte_count));
        check(static_cast<bool>(shared.front() == expected[offset]));
        check(static_cast<bool>(shared.back() == expected[offset + byte_count - 1]));
        MxFp4ByteBuffer copy = shared;
        copy.front() ^= 0xff;
        check(static_cast<bool>(shared.front() == expected[offset]));
        check(static_cast<bool>(copy.front() != shared.front()));

        auto truncated = MappedFileRange::open(
            path,
            file_size - 8,
            16);
        check(static_cast<bool>(!truncated));
        check(static_cast<bool>(truncated.error().code == ErrorCode::InvalidModel));
    }

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_safetensors_dense_mmap()
{
    const std::filesystem::path directory
        = create_unique_test_directory("ncnn_moe_safetensors_mmap_test_");
    const std::filesystem::path path = directory / "model.safetensors";
    std::string header
        = R"({"bf16":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"f32":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})";
    while (header.size() % 8 != 0)
        header.push_back(' ');
    const uint64_t header_size = header.size();
    const std::array<uint16_t, 2> bfloat_values = {
        float_to_bfloat16(1.0f),
        float_to_bfloat16(-2.0f)};
    const std::array<float, 2> float_values = {3.5f, -2.25f};
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(&header_size),
            sizeof(header_size));
        stream.write(
            header.data(),
            static_cast<std::streamsize>(header.size()));
        stream.write(
            reinterpret_cast<const char*>(bfloat_values.data()),
            sizeof(bfloat_values));
        stream.write(
            reinterpret_cast<const char*>(float_values.data()),
            sizeof(float_values));
    }

    {
        auto archive = SafetensorsArchive::open(directory);
        check(static_cast<bool>(archive));
        auto bfloat = archive.value().load_tensor("bf16");
        check(static_cast<bool>(bfloat));
        check(static_cast<bool>(bfloat.value().mapped_data));
        check(static_cast<bool>(bfloat.value().bfloat16_data.empty()));
        check(static_cast<bool>(bfloat.value().bfloat16_values().size() == 2));
        check_near(
            bfloat16_to_float(
                bfloat.value().bfloat16_values()[1]),
            -2.0f,
            0.0f);

        auto floating = archive.value().load_tensor("f32");
        check(static_cast<bool>(floating));
        check(static_cast<bool>(floating.value().mapped_data));
        check(static_cast<bool>(floating.value().float32_data.empty()));
        check(static_cast<bool>(floating.value().float32_values().size() == 2));
        check_near(
            floating.value().float32_values()[0],
            3.5f,
            0.0f);

        auto slice = archive.value().load_bfloat16_slice(
            "bf16",
            1,
            {1});
        check(static_cast<bool>(slice));
        check(static_cast<bool>(slice.value().mapped_data));
        check_near(
            bfloat16_to_float(
                slice.value().bfloat16_values()[0]),
            -2.0f,
            0.0f);
    }

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
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
    Mxfp4ExpertCache cache(34, 0, {}, ExpertCacheMemoryMapRanges);
    {
        auto first = cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(first));
        check(static_cast<bool>(!first.value().cache_hit));
        check(static_cast<bool>(first.value().bytes_read == 34));
        check(static_cast<bool>(first.value().gate_up->mxfp4_blocks.front() == 0));
        check(static_cast<bool>(first.value().down->mxfp4_blocks.front() == 16));
        check(static_cast<bool>(first.value().gate_up->mxfp4_scales.front() == 101));
        check(static_cast<bool>(first.value().down->mxfp4_scales.front() == 102));
    }
    {
        auto hit = cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(hit));
        check(static_cast<bool>(hit.value().cache_hit));
        check(static_cast<bool>(hit.value().bytes_read == 0));
    }
    {
        auto second = cache.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(second));
        check(static_cast<bool>(!second.value().cache_hit));
        check(static_cast<bool>(second.value().gate_up->mxfp4_blocks.front() == 32));
        check(static_cast<bool>(second.value().down->mxfp4_scales.front() == 104));
    }
    const ExpertCacheStatistics statistics = cache.statistics();
    check(static_cast<bool>(statistics.hits == 1));
    check(static_cast<bool>(statistics.misses == 2));
    check(static_cast<bool>(statistics.evictions == 1));
    check(static_cast<bool>(statistics.bytes_read == 68));
    check(static_cast<bool>(statistics.resident_bytes == 34));
    check(static_cast<bool>(statistics.mapped_ranges == 8));
    check(static_cast<bool>(statistics.mapped_bytes == statistics.bytes_read));

    auto victim = std::make_shared<TestExpertVictimCache>(68);
    Mxfp4ExpertCache tiered(
        34,
        1,
        victim,
        ExpertCacheMemoryMapRanges);
    {
        auto first = tiered.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(first));
        check(static_cast<bool>(first.value().gate_up->mxfp4_blocks.front() == 0));
    }
    {
        auto second = tiered.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(second));
        check(static_cast<bool>(second.value().gate_up->mxfp4_blocks.front() == 32));
    }
    {
        auto restored = tiered.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(restored));
        check(static_cast<bool>(!restored.value().cache_hit));
        check(static_cast<bool>(restored.value().gate_up->mxfp4_blocks.front() == 0));
        check(static_cast<bool>(restored.value().down->mxfp4_scales.front() == 102));
    }
    const ExpertCacheStatistics tiered_statistics = tiered.statistics();
    check(static_cast<bool>(tiered_statistics.misses == 3));
    check(static_cast<bool>(tiered_statistics.evictions == 2));
    check(static_cast<bool>(tiered_statistics.bytes_read == 68));
    check(static_cast<bool>(tiered_statistics.mapped_ranges == 8));
    check(static_cast<bool>(tiered_statistics.mapped_bytes == tiered_statistics.bytes_read));
    check(static_cast<bool>(tiered_statistics.victim.hits == 1));
    check(static_cast<bool>(tiered_statistics.victim.misses == 2));
    check(static_cast<bool>(tiered_statistics.victim.admissions == 2));

    if (NcnnLinearOperator::vulkan_device_count() > 0) {
        auto gpu_source = cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(gpu_source));
        auto gpu_victim = create_vulkan_expert_victim_cache(68);
        check(static_cast<bool>(gpu_victim));
        gpu_victim->admit(
            "gpu-roundtrip",
            gpu_source.value().gate_up,
            gpu_source.value().down);
        for (uint32_t attempt = 0;
             attempt < 2000
             && gpu_victim->statistics().stores == 0;
             ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        check(static_cast<bool>(gpu_victim->statistics().stores == 1));
        auto gpu_restored = gpu_victim->restore(
            "gpu-roundtrip",
            gate_zero,
            down_zero);
        check(static_cast<bool>(gpu_restored));
        check(static_cast<bool>(gpu_restored->gate_up->mxfp4_blocks.front() == 0));
        check(static_cast<bool>(gpu_restored->down->mxfp4_blocks.front() == 16));
        check(static_cast<bool>(gpu_restored->gate_up->mxfp4_scales.front() == 101));
        check(static_cast<bool>(gpu_restored->down->mxfp4_scales.front() == 102));
        const ExpertVictimCacheStatistics gpu_statistics
            = gpu_victim->statistics();
        check(static_cast<bool>(gpu_statistics.bytes_uploaded == 34));
        check(static_cast<bool>(gpu_statistics.bytes_downloaded == 34));
        check(static_cast<bool>(gpu_statistics.mapped_stores <= gpu_statistics.stores));
        check(static_cast<bool>(gpu_statistics.mapped_restores <= gpu_statistics.hits));
        check(static_cast<bool>(
            gpu_statistics.mapped_stores
            == gpu_statistics.mapped_restores));
    }

    Mxfp4ExpertCache concurrent(68, 2);
    check(static_cast<bool>(concurrent.request_pair(gate_zero, down_zero)));
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
    check(static_cast<bool>(first_ok));
    check(static_cast<bool>(second_ok));
    check(static_cast<bool>(first_hit != second_hit));
    check(static_cast<bool>(concurrent.statistics().misses == 1));
    check(static_cast<bool>(concurrent.statistics().hits == 1));
    check(static_cast<bool>(concurrent.statistics().queued_reads == 1));
    check(static_cast<bool>(concurrent.statistics().mapped_ranges == 0));
    check(static_cast<bool>(concurrent.statistics().mapped_bytes == 0));

    Mxfp4ExpertCache speculative(68, 1);
    check(static_cast<bool>(speculative.prefetch_pair(gate_zero, down_zero)));
    auto prefetched = speculative.acquire_pair(gate_zero, down_zero);
    check(static_cast<bool>(prefetched));
    check(static_cast<bool>(speculative.statistics().speculative_reads == 1));
    check(static_cast<bool>(speculative.statistics().queued_reads == 1));

    Mxfp4ExpertCache pressure(34, 1);
    auto pinned = pressure.acquire_pair(gate_zero, down_zero);
    check(static_cast<bool>(pinned));
    check(static_cast<bool>(pressure.prefetch_pair(gate_one, down_one)));
    check(static_cast<bool>(pressure.statistics().speculative_reads == 0));
    auto exhausted = pressure.acquire_pair(gate_one, down_one);
    check(static_cast<bool>(!exhausted));
    check(static_cast<bool>(exhausted.error().code == ErrorCode::InvalidArgument));

    const TensorData truncated_gate = file_backed(60, 0);
    Mxfp4ExpertCache retryable(34, 1);
    auto failed_read = retryable.acquire_pair(truncated_gate, down_zero);
    check(static_cast<bool>(!failed_read));
    check(static_cast<bool>(failed_read.error().code == ErrorCode::IoError));
    auto retried_read = retryable.acquire_pair(truncated_gate, down_zero);
    check(static_cast<bool>(!retried_read));
    check(static_cast<bool>(retried_read.error().code == ErrorCode::IoError));
    check(static_cast<bool>(retryable.statistics().misses == 2));
    check(static_cast<bool>(retryable.statistics().resident_bytes == 0));

    Mxfp4ExpertCache undersized(33);
    auto rejected = undersized.acquire_pair(gate_zero, down_zero);
    check(static_cast<bool>(!rejected));
    check(static_cast<bool>(rejected.error().code == ErrorCode::InvalidArgument));

    const std::filesystem::path large_blocks_path = directory / "large_blocks.bin";
    const std::filesystem::path large_scales_path = directory / "large_scales.bin";
    constexpr uint64_t large_rows = 393216;
    constexpr uint64_t large_blocks_per_tensor = large_rows * 16;
    constexpr uint64_t large_scales_per_tensor = large_rows;
    {
        std::vector<uint8_t> blocks(large_blocks_per_tensor * 2);
        for (size_t index = 0; index < blocks.size(); ++index)
            blocks[index] = static_cast<uint8_t>((index * 13) % 251);
        std::ofstream stream(large_blocks_path, std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(blocks.data()),
            static_cast<std::streamsize>(blocks.size()));
    }
    {
        std::vector<uint8_t> scales(large_scales_per_tensor * 2);
        for (size_t index = 0; index < scales.size(); ++index)
            scales[index] = static_cast<uint8_t>((index * 7) % 239);
        std::ofstream stream(large_scales_path, std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(scales.data()),
            static_cast<std::streamsize>(scales.size()));
    }

    auto large_file_backed = [&](uint64_t block_offset, uint64_t scale_offset) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {large_rows, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = large_blocks_path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_bytes = large_blocks_per_tensor;
        storage->scales_path = large_scales_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_bytes = large_scales_per_tensor;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    const TensorData large_gate = large_file_backed(0, 0);
    const TensorData large_down = large_file_backed(
        large_blocks_per_tensor,
        large_scales_per_tensor);
    const uint64_t large_pair_bytes
        = (large_blocks_per_tensor + large_scales_per_tensor) * 2;
    Mxfp4ExpertCache large_reads(
        large_pair_bytes,
        2,
        {},
        true);
    auto large_pair = large_reads.acquire_pair(large_gate, large_down);
    check(static_cast<bool>(large_pair));
    check(static_cast<bool>(large_pair.value().bytes_read == large_pair_bytes));
    check(static_cast<bool>(large_reads.statistics().mapped_ranges == 4));
    check(static_cast<bool>(large_reads.statistics().mapped_bytes == large_pair_bytes));
    check(static_cast<bool>(large_pair.value().gate_up->mxfp4_blocks.front() == 0));
    check(static_cast<bool>(
        large_pair.value().gate_up->mxfp4_blocks.back()
        == static_cast<uint8_t>(((large_blocks_per_tensor - 1) * 13) % 251)));
    check(static_cast<bool>(
        large_pair.value().down->mxfp4_scales.back()
        == static_cast<uint8_t>(
            ((large_scales_per_tensor * 2 - 1) * 7) % 239)));

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_cpu_topology_parsing_and_partitioning()
{
    check(static_cast<bool>(
        parse_linux_cpu_list("0-3,8,10-11")
        == std::vector<uint32_t>({0, 1, 2, 3, 8, 10, 11})));
    check(static_cast<bool>(
        parse_linux_cpu_list("4,2-4,2")
        == std::vector<uint32_t>({2, 3, 4})));
    check(static_cast<bool>(parse_linux_cpu_list("3-1").empty()));
    check(static_cast<bool>(parse_linux_cpu_list("1,,2").empty()));
    check(static_cast<bool>(parse_linux_cpu_list("1,").empty()));

    CpuTopology flat;
    flat.allowed_cpus = {2, 4, 7, 9, 12};
    const std::vector<std::vector<uint32_t> > flat_partitions
        = partition_cpu_topology(flat, 2);
    check(static_cast<bool>(flat_partitions.size() == 2));
    check(static_cast<bool>(flat_partitions[0] == std::vector<uint32_t>({2, 4, 7})));
    check(static_cast<bool>(flat_partitions[1] == std::vector<uint32_t>({9, 12})));

    CpuTopology numa;
    numa.allowed_cpus = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    numa.numa_nodes = {
        {0, 1, 2, 3},
        {4, 5, 6, 7, 8, 9, 10, 11},
    };
    const std::vector<std::vector<uint32_t> > numa_partitions
        = partition_cpu_topology(numa, 4);
    check(static_cast<bool>(numa_partitions.size() == 4));
    std::vector<uint32_t> partitioned_cpus;
    for (const std::vector<uint32_t>& partition : numa_partitions) {
        check(static_cast<bool>(!partition.empty()));
        const bool first_node = partition.front() < 4;
        for (uint32_t cpu : partition)
            check(static_cast<bool>((cpu < 4) == first_node));
        partitioned_cpus.insert(
            partitioned_cpus.end(), partition.begin(), partition.end());
    }
    std::sort(partitioned_cpus.begin(), partitioned_cpus.end());
    check(static_cast<bool>(partitioned_cpus == numa.allowed_cpus));
}

void test_cross_session_batch_scheduler()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    auto first = runtime.create_session(model.value());
    auto second = runtime.create_session(model.value());
    check(static_cast<bool>(first));
    check(static_cast<bool>(second));

    SchedulerOptions options;
    options.worker_count = 2;
    auto scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(scheduler));
    auto future = scheduler.value()->submit_decode({
        {first.value(), 0},
        {second.value(), 1},
    });
    std::vector<Result<DecodeResult> > results = future.get();
    check(static_cast<bool>(results.size() == 2));
    check(static_cast<bool>(results[0]));
    check(static_cast<bool>(results[1]));
    check(static_cast<bool>(first.value()->sequence_length() == 1));
    check(static_cast<bool>(second.value()->sequence_length() == 1));

    auto ordered_session = runtime.create_session(model.value());
    auto reference_session = runtime.create_session(model.value());
    check(static_cast<bool>(ordered_session));
    check(static_cast<bool>(reference_session));
    auto ordered_first = scheduler.value()->submit_decode({
        {ordered_session.value(), 0},
    });
    auto ordered_second = scheduler.value()->submit_decode({
        {ordered_session.value(), 1},
    });
    auto reference_first = reference_session.value()->decode(0);
    auto reference_second = reference_session.value()->decode(1);
    check(static_cast<bool>(reference_first));
    check(static_cast<bool>(reference_second));
    std::vector<Result<DecodeResult> > ordered_first_result = ordered_first.get();
    std::vector<Result<DecodeResult> > ordered_second_result = ordered_second.get();
    check(static_cast<bool>(ordered_first_result[0]));
    check(static_cast<bool>(ordered_second_result[0]));
    check(static_cast<bool>(ordered_first_result[0].value().sequence_length == 1));
    check(static_cast<bool>(ordered_second_result[0].value().sequence_length == 2));
    for (size_t index = 0;
         index < reference_second.value().logits.values.size();
         ++index) {
        check_near(
            ordered_second_result[0].value().logits.values[index],
            reference_second.value().logits.values[index],
            1e-5f);
    }

    auto duplicate_future = scheduler.value()->submit_decode({
        {first.value(), 0},
        {first.value(), 1},
    });
    std::vector<Result<DecodeResult> > duplicate_results = duplicate_future.get();
    check(static_cast<bool>(!duplicate_results[0]));
    check(static_cast<bool>(!duplicate_results[1]));
    check(static_cast<bool>(first.value()->sequence_length() == 1));
    const SchedulerStatistics statistics = scheduler.value()->statistics();
    check(static_cast<bool>(statistics.worker_count == 2));
    check(static_cast<bool>(statistics.expert_threads_per_worker >= 1));
    check(static_cast<bool>(statistics.submitted_batches == 4));
    check(static_cast<bool>(statistics.submitted_requests == 6));
    check(static_cast<bool>(statistics.completed_requests == 6));
    check(static_cast<bool>(statistics.rejected_requests == 2));
    check(static_cast<bool>(statistics.max_batch_size == 2));
    check(static_cast<bool>(statistics.max_in_flight >= 2));

    SchedulerOptions mismatched_affinity;
    mismatched_affinity.worker_count = 2;
    mismatched_affinity.worker_cpu_sets = {{0}};
    auto mismatched_scheduler = runtime.create_scheduler(mismatched_affinity);
    check(static_cast<bool>(!mismatched_scheduler));
    check(static_cast<bool>(mismatched_scheduler.error().code == ErrorCode::InvalidArgument));

    SchedulerOptions empty_affinity;
    empty_affinity.worker_cpu_sets = {{}};
    auto empty_affinity_scheduler = runtime.create_scheduler(empty_affinity);
    check(static_cast<bool>(!empty_affinity_scheduler));
    check(static_cast<bool>(empty_affinity_scheduler.error().code == ErrorCode::InvalidArgument));
#if !defined(__linux__)
    SchedulerOptions unsupported_affinity;
    unsupported_affinity.worker_cpu_sets = {{0}};
    auto unsupported_scheduler = runtime.create_scheduler(unsupported_affinity);
    check(static_cast<bool>(!unsupported_scheduler));
    check(static_cast<bool>(unsupported_scheduler.error().code == ErrorCode::UnsupportedModel));
#endif

    if (has_flag(
            runtime.capabilities().flags,
            RuntimeCapabilityVulkanCpuMix)) {
        AttentionPackage attention_package;
        RuntimeOptions hybrid_options;
        hybrid_options.hybrid_mode = HybridMode::HybridExperts;
        RuntimeOptions cpu_options;
        cpu_options.hybrid_mode = HybridMode::CpuOnly;
        auto hybrid_model = runtime.load_model(
            attention_package.path(), hybrid_options);
        auto cpu_model = runtime.load_model(attention_package.path(), cpu_options);
        check(static_cast<bool>(hybrid_model));
        check(static_cast<bool>(cpu_model));

        SchedulerOptions pipeline_options;
        pipeline_options.worker_count = 4;
        auto pipeline_scheduler = runtime.create_scheduler(pipeline_options);
        check(static_cast<bool>(pipeline_scheduler));
        std::vector<SessionPtr> hybrid_sessions;
        std::vector<SessionPtr> cpu_sessions;
        for (uint32_t index = 0; index < 4; ++index) {
            auto hybrid_session = runtime.create_session(hybrid_model.value());
            auto cpu_session = runtime.create_session(cpu_model.value());
            check(static_cast<bool>(hybrid_session));
            check(static_cast<bool>(cpu_session));
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
            check(static_cast<bool>(pipeline_results.size() == hybrid_sessions.size()));
            for (uint32_t session_index = 0;
                 session_index < hybrid_sessions.size();
                 ++session_index) {
                auto cpu_result = cpu_sessions[session_index]->decode(
                    static_cast<int32_t>((round + session_index) % 2));
                check(static_cast<bool>(pipeline_results[session_index]));
                check(static_cast<bool>(cpu_result));
                check(static_cast<bool>(pipeline_results[session_index].value().sequence_length
                                        == round + 1));
                check(static_cast<bool>(
                    pipeline_results[session_index].value().logits.values.size()
                    == cpu_result.value().logits.values.size()));
                for (size_t logit = 0;
                     logit < cpu_result.value().logits.values.size();
                     ++logit) {
                    check_near(
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
            check(static_cast<bool>(session_statistics.vulkan_attention_blocks == pipeline_rounds));
            check(static_cast<bool>(
                session_statistics.vulkan_compute_submissions
                == pipeline_rounds * 2));
            check(static_cast<bool>(
                session_statistics.vulkan_staging_slot_acquisitions
                == pipeline_rounds * 2));
        }
        const SchedulerStatistics pipeline_statistics
            = pipeline_scheduler.value()->statistics();
        check(static_cast<bool>(pipeline_statistics.completed_requests == pipeline_rounds * 4));
        check(static_cast<bool>(pipeline_statistics.max_in_flight >= 4));
    }
}

void test_invalid_token_is_transactional()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));

    auto invalid = session.value()->decode(4);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));
    check(static_cast<bool>(session.value()->sequence_length() == 0));
    check(static_cast<bool>(session.value()->statistics().decode_tokens == 0));
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 0));

    const std::vector<int32_t> empty;
    auto empty_prefill = session.value()->prefill(empty);
    check(static_cast<bool>(!empty_prefill));
    check(static_cast<bool>(empty_prefill.error().code == ErrorCode::InvalidArgument));
}

void test_chunked_prefill_matches_single_batch()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));

    SessionOptions single_batch_options;
    single_batch_options.prefill_chunk_size = 0;
    auto single_batch_session = runtime.create_session(model.value(), single_batch_options);
    check(static_cast<bool>(single_batch_session));

    SessionOptions chunked_options;
    chunked_options.prefill_chunk_size = 2;
    auto chunked_session = runtime.create_session(model.value(), chunked_options);
    check(static_cast<bool>(chunked_session));

    const std::vector<int32_t> prompt = {0, 1, 2};
    auto single_batch = single_batch_session.value()->prefill(prompt);
    auto chunked = chunked_session.value()->prefill(prompt);
    check(static_cast<bool>(single_batch));
    check(static_cast<bool>(chunked));
    check(static_cast<bool>(chunked.value().processed_tokens == prompt.size()));
    check(static_cast<bool>(chunked_session.value()->sequence_length() == prompt.size()));
    check(static_cast<bool>(chunked_session.value()->statistics().prefill_tokens == prompt.size()));
    check(static_cast<bool>(chunked.value().logits.values.size() == single_batch.value().logits.values.size()));
    for (size_t index = 0; index < single_batch.value().logits.values.size(); ++index) {
        check_near(
            chunked.value().logits.values[index],
            single_batch.value().logits.values[index],
            1e-5f);
    }
}

void test_topk_selected_weight_normalization_and_combine()
{
    WeightedTopKPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));

    auto result = session.value()->decode(0);
    check(static_cast<bool>(result));

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

    check_near(result.value().logits.values[0], hidden_x / final_scale, 1e-5f);
    check_near(result.value().logits.values[1], hidden_y / final_scale, 1e-5f);
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 2));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 2));
    check(static_cast<bool>(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({1, 1, 0})));
    if (has_flag(
            runtime.capabilities().flags,
            RuntimeCapabilityOpenmpExpertParallelism)) {
        check(static_cast<bool>(session.value()->statistics().expert_parallel_tasks == 2));
    }
}

void test_int8_expert_linear()
{
    Int8ExpertPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    check(static_cast<bool>(model.value()->descriptor().layers[0].ffn.moe.expert_weight_dtype == DType::Int8));

    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));
    auto result = session.value()->decode(0);
    check(static_cast<bool>(result));

    const float normalized_input = 1.0f / std::sqrt(0.5f + 1e-5f);
    const float hidden = 1.0f + normalized_input;
    const float expected = hidden / std::sqrt(hidden * hidden / 2.0f + 1e-5f);
    check_near(result.value().logits.values[0], expected, 1e-5f);
    check_near(result.value().logits.values[1], 0.0f, 1e-5f);
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 1));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 1));
}

void test_invalid_int8_scale_is_rejected()
{
    Int8ExpertPackage package(true);
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(!model));
    check(static_cast<bool>(model.error().code == ErrorCode::InvalidModel));
}

void test_attention_kv_cache_and_reset()
{
    AttentionPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    check(static_cast<bool>(has_flag(
        model.value()->descriptor().layers[0].flags,
        LayerDescriptorAttention)));

    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));
    const std::vector<int32_t> prompt = {0};
    auto prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill));
    auto cached_decode = session.value()->decode(1);
    check(static_cast<bool>(cached_decode));
    check(static_cast<bool>(cached_decode.value().logits.values[0] > 0.1f));

    check(static_cast<bool>(session.value()->reset()));
    auto uncached_decode = session.value()->decode(1);
    check(static_cast<bool>(uncached_decode));
    check_near(uncached_decode.value().logits.values[0], 0.0f, 1e-6f);
}

void test_bfloat16_ring_kv_cache()
{
    AttentionPackage float32_package;
    AttentionPackage bfloat16_package(true);
    TestRuntime runtime;
    RuntimeOptions cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto float32_model = runtime.load_model(float32_package.path(), cpu_options);
    auto bfloat16_model = runtime.load_model(bfloat16_package.path(), cpu_options);
    check(static_cast<bool>(float32_model));
    check(static_cast<bool>(bfloat16_model));
    check(static_cast<bool>(bfloat16_model.value()->descriptor().kv_cache_dtype == DType::BFloat16));

    auto float32_session = runtime.create_session(float32_model.value());
    auto bfloat16_session = runtime.create_session(bfloat16_model.value());
    check(static_cast<bool>(float32_session));
    check(static_cast<bool>(bfloat16_session));

    const std::vector<int32_t> prompt = {0, 1, 0, 1, 0, 1, 0, 1};
    check(static_cast<bool>(float32_session.value()->prefill(prompt)));
    check(static_cast<bool>(bfloat16_session.value()->prefill(prompt)));
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_logical_bytes
                            == float32_session.value()->statistics().kv_cache_logical_bytes / 2));
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_allocated_bytes
                            == float32_session.value()->statistics().kv_cache_allocated_bytes / 2));

    for (uint32_t index = 0; index < 16; ++index)
        check(static_cast<bool>(bfloat16_session.value()->decode(static_cast<int32_t>(index % 2))));
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_allocated_bytes
                            <= bfloat16_session.value()->statistics().kv_cache_logical_bytes * 16));
}

void test_attention_graph_without_bias_or_sink()
{
    AttentionPackage package(false, 0, false, false);
    std::ifstream manifest_stream(package.path() / "config.json");
    const std::string manifest_json{
        std::istreambuf_iterator<char>(manifest_stream),
        std::istreambuf_iterator<char>()};
    ModelPackage model_package;
    model_package.root = package.path();
    model_package.manifest.model_type = "test_moe";
    model_package.manifest.raw_json = manifest_json;
    FixtureModelAdapter adapter;
    auto descriptor = adapter.parse_model(model_package);
    check(static_cast<bool>(descriptor));
    auto mapping = adapter.map_weights(model_package, descriptor.value());
    check(static_cast<bool>(mapping));
    descriptor.value().layers[0].flags
        &= ~(LayerDescriptorAttention | LayerDescriptorMoe);
    ModelCompiler compiler;
    auto graph_driven_model = compiler.compile(
        std::move(descriptor).value(),
        std::move(mapping).value(),
        HybridMode::CpuOnly);
    check(static_cast<bool>(graph_driven_model));
    check(static_cast<bool>(has_flag(
        graph_driven_model.value().layers[0].flags,
        CompiledLayerAttention)));

    TestRuntime runtime;
    RuntimeOptions cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto cpu_model = runtime.load_model(package.path(), cpu_options);
    check(static_cast<bool>(cpu_model));
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
    check(static_cast<bool>(layer.nodes.size() == expected_nodes.size()));
    for (size_t index = 0; index < expected_nodes.size(); ++index)
        check(static_cast<bool>(layer.nodes[index].type == expected_nodes[index]));
    check(static_cast<bool>(!has_flag(layer.attention.flags, AttentionDescriptorBias)));
    check(static_cast<bool>(!has_flag(layer.attention.flags, AttentionDescriptorSinks)));
    const CompiledLayerPlan& cpu_plan
        = cpu_model.value()->execution_plan()[0];
    check(static_cast<bool>(cpu_plan.nodes.size() == expected_nodes.size()));
    for (const CompiledNodePlan& node : cpu_plan.nodes)
        check(static_cast<bool>(node.backend == ExecutionBackend::Cpu));

    auto cpu_session = runtime.create_session(cpu_model.value());
    check(static_cast<bool>(cpu_session));
    const std::vector<int32_t> prompt = {0, 1, 0};
    auto cpu_prefill = cpu_session.value()->prefill(prompt);
    check(static_cast<bool>(cpu_prefill));

    if (has_flag(
            runtime.capabilities().flags,
            RuntimeCapabilityVulkanAttention)) {
        RuntimeOptions hybrid_options;
        hybrid_options.hybrid_mode = HybridMode::HybridExperts;
        auto hybrid_model = runtime.load_model(package.path(), hybrid_options);
        check(static_cast<bool>(hybrid_model));
        const CompiledLayerPlan& hybrid_plan
            = hybrid_model.value()->execution_plan()[0];
        check(static_cast<bool>(hybrid_plan.nodes.size() == expected_nodes.size()));
        for (size_t node_index = 0;
             node_index < hybrid_plan.nodes.size();
             ++node_index) {
            check(static_cast<bool>(
                hybrid_plan.nodes[node_index].backend
                == (node_index < 5
                        ? ExecutionBackend::Vulkan
                        : ExecutionBackend::Cpu)));
        }
        const ExecutionGraph& hybrid_graph
            = hybrid_model.value()->execution_graph();
        size_t vulkan_attention_nodes = 0;
        size_t vulkan_lm_head_nodes = 0;
        for (const ExecutionNode& node : hybrid_graph.nodes) {
            if (node.type == ExecutionNodeType::Attention
                && node.backend == ExecutionBackend::Vulkan) {
                ++vulkan_attention_nodes;
            }
            if (node.type == ExecutionNodeType::LmHead
                && node.backend == ExecutionBackend::Vulkan) {
                ++vulkan_lm_head_nodes;
            }
            if (node.type == ExecutionNodeType::Expert)
                check(static_cast<bool>(node.backend == ExecutionBackend::Cpu));
        }
        check(static_cast<bool>(vulkan_attention_nodes == 1));
        check(static_cast<bool>(vulkan_lm_head_nodes == 1));
        auto hybrid_session = runtime.create_session(hybrid_model.value());
        check(static_cast<bool>(hybrid_session));
        auto hybrid_prefill = hybrid_session.value()->prefill(prompt);
        check(static_cast<bool>(hybrid_prefill));
        check(static_cast<bool>(hybrid_session.value()->statistics().vulkan_attention_blocks == 1));
        check(static_cast<bool>(hybrid_prefill.value().logits.values.size()
                                == cpu_prefill.value().logits.values.size()));
        for (size_t index = 0;
             index < cpu_prefill.value().logits.values.size();
             ++index) {
            check_near(
                hybrid_prefill.value().logits.values[index],
                cpu_prefill.value().logits.values[index],
                1e-4f);
        }
    }
}

void test_moe_ir_execution_graph_and_scheduler()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    RuntimeOptions options;
    options.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), options);
    check(static_cast<bool>(model));
    check(static_cast<bool>(model.value()->ir().model_type == "test_moe"));

    const ExecutionGraph& graph = model.value()->execution_graph();
    check(static_cast<bool>(graph.validate()));
    check(static_cast<bool>(graph.nodes.size() == 8));
    const std::vector<ExecutionNodeType> expected_types = {
        ExecutionNodeType::TokenEmbedding,
        ExecutionNodeType::Router,
        ExecutionNodeType::ExpertDispatch,
        ExecutionNodeType::Expert,
        ExecutionNodeType::Expert,
        ExecutionNodeType::Combine,
        ExecutionNodeType::FinalNorm,
        ExecutionNodeType::LmHead,
    };
    for (size_t index = 0; index < expected_types.size(); ++index) {
        check(static_cast<bool>(graph.nodes[index].id == index));
        check(static_cast<bool>(graph.nodes[index].type == expected_types[index]));
        check(static_cast<bool>(graph.nodes[index].backend == ExecutionBackend::Cpu));
    }
    check(static_cast<bool>(has_flag(graph.nodes[3].flags, ExecutionNodeConditional)));
    check(static_cast<bool>(graph.nodes[3].expert_id == 0));
    check(static_cast<bool>(has_flag(graph.nodes[4].flags, ExecutionNodeConditional)));
    check(static_cast<bool>(graph.nodes[4].expert_id == 1));
    check(static_cast<bool>(graph.nodes[5].dependencies.size() == 2));
    check(static_cast<bool>(graph.nodes[5].dependencies[0] == 3));
    check(static_cast<bool>(graph.nodes[5].dependencies[1] == 4));

    const ExecutionSchedule& compiled_schedule
        = model.value()->execution_schedule();
    check(static_cast<bool>(compiled_schedule.waves.size() == 7));
    check(static_cast<bool>(compiled_schedule.waves[3].nodes.size() == 2));
    check(static_cast<bool>(
        graph.find(compiled_schedule.waves[3].nodes[0])->type
        == ExecutionNodeType::Expert));
    check(static_cast<bool>(
        graph.find(compiled_schedule.waves[3].nodes[1])->type
        == ExecutionNodeType::Expert));

    MoeScheduler scheduler;
    auto rescheduled = scheduler.schedule(graph);
    check(static_cast<bool>(rescheduled));
    check(static_cast<bool>(rescheduled.value().waves.size() == compiled_schedule.waves.size()));

    ExecutionGraph cyclic;
    cyclic.nodes = {
        {0, ExecutionNodeType::Router, ExecutionBackend::Cpu, 0, invalid_execution_expert_id, false, "router", {1}},
        {1, ExecutionNodeType::Combine, ExecutionBackend::Cpu, 0, invalid_execution_expert_id, false, "combine", {0}},
    };
    auto invalid_schedule = scheduler.schedule(cyclic);
    check(static_cast<bool>(!invalid_schedule));
    check(static_cast<bool>(invalid_schedule.error().code == ErrorCode::InvalidModel));
}

void test_expert_dispatcher_groups_routes()
{
    ExpertDispatchOptions options;
    options.expert_count = 6;
    options.top_k = 1;
    const std::vector<float> logits = {
        0.0f,
        5.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        5.0f,
        0.0f,
        4.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        4.0f,
    };
    ExpertDispatcher dispatcher;
    auto dispatch = dispatcher.dispatch(logits, 4, options);
    check(static_cast<bool>(dispatch));
    check(static_cast<bool>(dispatch.value().assignment_count == 4));
    check(static_cast<bool>(dispatch.value().batches.size() == 2));
    check(static_cast<bool>(dispatch.value().batches[0].expert_id == 1));
    check(static_cast<bool>(dispatch.value().batches[0].routes.size() == 2));
    check(static_cast<bool>(dispatch.value().batches[0].routes[0].token_index == 0));
    check(static_cast<bool>(dispatch.value().batches[0].routes[1].token_index == 2));
    check_near(dispatch.value().batches[0].routes[0].weight, 1.0f, 1e-6f);
    check(static_cast<bool>(dispatch.value().batches[1].expert_id == 5));
    check(static_cast<bool>(dispatch.value().batches[1].routes.size() == 2));
    check(static_cast<bool>(dispatch.value().batches[1].routes[0].token_index == 1));
    check(static_cast<bool>(dispatch.value().batches[1].routes[1].token_index == 3));

    options.expert_count = 3;
    options.top_k = 2;
    const std::vector<float> weighted_logits = {2.0f, 1.0f, 0.0f};
    auto weighted = dispatcher.dispatch(
        weighted_logits,
        1,
        options);
    check(static_cast<bool>(weighted));
    check(static_cast<bool>(weighted.value().assignment_count == 2));
    check(static_cast<bool>(weighted.value().batches.size() == 2));
    check_near(
        weighted.value().batches[0].routes[0].weight
            + weighted.value().batches[1].routes[0].weight,
        1.0f,
        1e-6f);

    const std::vector<float> invalid_logits = {1.0f, 2.0f};
    auto invalid = dispatcher.dispatch(invalid_logits, 1, options);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));
}

static MoeIR gpt_oss_memory_ir(
    uint32_t layer_count,
    uint32_t expert_count)
{
    MoeIR ir;
    ir.model_type = "gpt_oss";
    ir.vocabulary_size = 201088;
    ir.hidden_size = 2880;
    ir.intermediate_size = 2880;
    ir.layer_count = layer_count;
    ir.attention_head_count = 64;
    ir.kv_head_count = 8;
    ir.head_dimension = 64;
    ir.expert_count = expert_count;
    ir.experts_per_token = 4;
    ir.activation_dtype = DType::BFloat16;
    ir.kv_cache_dtype = DType::BFloat16;
    ir.layers.resize(layer_count);
    for (LayerDescriptor& layer : ir.layers) {
        layer.flags |= LayerDescriptorAttention;
        layer.attention.head_count = 64;
        layer.attention.kv_head_count = 8;
        layer.attention.head_dimension = 64;
        layer.attention.flags |= AttentionDescriptorBias
                                 | AttentionDescriptorSinks;
        layer.ffn.moe.expert_count = expert_count;
        layer.ffn.moe.top_k = 4;
        layer.ffn.moe.intermediate_size = 2880;
        layer.ffn.moe.expert_weight_dtype = DType::MxFp4;
        layer.ffn.moe.flags |= MoeDescriptorRouterBias
                               | MoeDescriptorProjectionBias;
    }
    return ir;
}

void test_automatic_expert_memory_planning()
{
    static constexpr uint64_t gibibyte
        = 1024ull * 1024ull * 1024ull;
    RuntimeOptions options;
    const uint64_t physical_memory = 32 * gibibyte;

    const MoeIR small = gpt_oss_memory_ir(24, 32);
    auto small_plan = plan_model_memory(small, options, physical_memory);
    check(static_cast<bool>(small_plan));
    check(static_cast<bool>(small_plan.value().selected_mode == ExpertMemoryMode::Eager));
    check(static_cast<bool>(!has_flag(
        small_plan.value().flags,
        ModelMemoryFileBackedExperts)));
    check(static_cast<bool>(small_plan.value().estimated_expert_bytes < 11 * gibibyte));

    const MoeIR large = gpt_oss_memory_ir(36, 128);
    auto large_plan = plan_model_memory(large, options, physical_memory);
    check(static_cast<bool>(large_plan));
    check(static_cast<bool>(large_plan.value().selected_mode == ExpertMemoryMode::OnDemand));
    check(static_cast<bool>(has_flag(
        large_plan.value().flags,
        ModelMemoryFileBackedExperts)));
    check(static_cast<bool>(large_plan.value().host_memory_budget_bytes == 24 * gibibyte));
    check(static_cast<bool>(large_plan.value().estimated_dense_bytes == 4334742144ull));
    check(static_cast<bool>(large_plan.value().expert_pair_bytes == 13219200ull));
    check(static_cast<bool>(large_plan.value().estimated_expert_bytes == 60914073600ull));
    check(static_cast<bool>(
        large_plan.value().expert_cache_bytes
        == 20 * gibibyte
               - large_plan.value().estimated_dense_bytes));
    check(static_cast<bool>(
        large_plan.value().expert_cache_bytes
        >= large_plan.value().minimum_active_expert_bytes));

    RuntimeOptions undersized;
    undersized.expert_cache_bytes = 32 * 1024 * 1024;
    auto invalid = plan_model_memory(large, undersized, physical_memory);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));

    RuntimeOptions over_budget;
    over_budget.host_memory_budget_bytes = 33 * gibibyte;
    invalid = plan_model_memory(large, over_budget, physical_memory);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));

    RuntimeOptions oversized_cache;
    oversized_cache.host_memory_budget_bytes = 24 * gibibyte;
    oversized_cache.expert_cache_bytes = 21 * gibibyte;
    invalid = plan_model_memory(large, oversized_cache, physical_memory);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));
}

void test_sampling_and_streaming_generation()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));

    SessionOptions session_options;
    session_options.sampling_seed = 1234;
    auto session = runtime.create_session(model.value(), session_options);
    check(static_cast<bool>(session));

    LogitsOutput logits;
    logits.values = {1.0f, 2.0f, 3.0f};
    SamplingOptions greedy_options;
    greedy_options.temperature = 0.0f;
    auto greedy = session.value()->sample(logits, greedy_options);
    check(static_cast<bool>(greedy));
    check(static_cast<bool>(greedy.value().token_id == 2));
    check_near(greedy.value().probability, 1.0f, 1e-6f);

    SamplingOptions top_k_options;
    top_k_options.top_k = 1;
    auto top_k = session.value()->sample(logits, top_k_options);
    check(static_cast<bool>(top_k));
    check(static_cast<bool>(top_k.value().token_id == 2));

    SamplingOptions invalid_options;
    invalid_options.top_p = 0.0f;
    auto invalid = session.value()->sample(logits, invalid_options);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));

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
    check(static_cast<bool>(generated));
    check(static_cast<bool>(generated.value().tokens.size() == 3));
    check(static_cast<bool>(streamed.size() == 3));
    check(static_cast<bool>(generated.value().tokens[0].text
                            == "<" + std::to_string(generated.value().tokens[0].token_id) + ">"));
    check(static_cast<bool>(session.value()->sequence_length() == 3));

    auto stopped_session = runtime.create_session(model.value(), session_options);
    check(static_cast<bool>(stopped_session));
    auto stopped = stopped_session.value()->generate(
        std::vector<int32_t>{0},
        generation_options,
        [](const StreamToken&) { return false; });
    check(static_cast<bool>(stopped));
    check(static_cast<bool>(stopped.value().stopped_by_callback));
    check(static_cast<bool>(stopped.value().tokens.size() == 1));
    check(static_cast<bool>(stopped_session.value()->sequence_length() == 1));
}

void test_loader_reports_adapter_and_weight_errors()
{
    {
        TemporaryModelPackage package;
        package.write_manifest("unknown_family");
        TestRuntime runtime;
        auto model = runtime.load_model(package.path());
        check(static_cast<bool>(!model));
        check(static_cast<bool>(model.error().code == ErrorCode::UnsupportedModel));
    }

    {
        TemporaryModelPackage package;
        package.truncate_weights();
        TestRuntime runtime;
        auto model = runtime.load_model(package.path() / "config.json");
        check(static_cast<bool>(!model));
        check(static_cast<bool>(model.error().code == ErrorCode::InvalidModel));
    }

    {
        TemporaryModelPackage package;
        TestRuntime runtime;
        RuntimeOptions options;
        options.expert_memory_mode = ExpertMemoryMode::Eager;
        options.expert_gpu_cache_bytes = 64 * 1024 * 1024;
        auto model = runtime.load_model(package.path(), options);
        check(static_cast<bool>(!model));
        check(static_cast<bool>(model.error().code == ErrorCode::InvalidArgument));
    }
}

void test_backend_capabilities_and_hybrid_execution()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    check(static_cast<bool>(has_flag(
        runtime.capabilities().flags,
        RuntimeCapabilityCpuExecution)));
    check(static_cast<bool>(has_flag(
                                runtime.capabilities().flags,
                                RuntimeCapabilityVulkanCpuPrefetch)
                            == has_flag(
                                runtime.capabilities().flags,
                                RuntimeCapabilityVulkanCpuMix)));

    RuntimeOptions automatic_options;
    automatic_options.hybrid_mode = HybridMode::Auto;
    auto automatic_model = runtime.load_model(package.path(), automatic_options);
    check(static_cast<bool>(automatic_model));
    check(static_cast<bool>(automatic_model.value()->hybrid_mode()
                            == (has_flag(
                                    runtime.capabilities().flags,
                                    RuntimeCapabilityVulkanCpuMix)
                                    ? HybridMode::HybridExperts
                                    : HybridMode::CpuOnly)));
    auto automatic_session = runtime.create_session(automatic_model.value());
    check(static_cast<bool>(automatic_session));
    const std::vector<int32_t> packed_prompt = {0, 1, 2, 3};
    auto automatic_prefill = automatic_session.value()->prefill(packed_prompt);
    check(static_cast<bool>(automatic_prefill));
    if (has_flag(
            runtime.capabilities().flags,
            RuntimeCapabilityVulkanCpuMix)) {
        const SessionStatistics& statistics = automatic_session.value()->statistics();
        check(static_cast<bool>(statistics.vulkan_linear_dispatches == 1));
        check(static_cast<bool>(statistics.vulkan_compute_submissions == 1));
        check(static_cast<bool>(statistics.vulkan_batch_uploads == 1));
        check(static_cast<bool>(statistics.vulkan_batch_downloads == 1));
        check(static_cast<bool>(statistics.vulkan_auxiliary_uploads == 0));
        check(static_cast<bool>(statistics.vulkan_auxiliary_upload_bytes == 0));
        check(static_cast<bool>(statistics.vulkan_staging_slot_resizes
                                    + statistics.vulkan_staging_slot_reuses
                                == 2));
        check(static_cast<bool>(statistics.vulkan_staging_slot_acquisitions == 1));
    }
    else
        check(static_cast<bool>(automatic_session.value()->statistics().vulkan_linear_dispatches == 0));

    RuntimeOptions cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto cpu_model = runtime.load_model(package.path(), cpu_options);
    check(static_cast<bool>(cpu_model));
    auto cpu_session = runtime.create_session(cpu_model.value());
    check(static_cast<bool>(cpu_session));
    auto cpu_prefill = cpu_session.value()->prefill(packed_prompt);
    check(static_cast<bool>(cpu_prefill));
    check(static_cast<bool>(cpu_prefill.value().logits.values.size() == automatic_prefill.value().logits.values.size()));
    for (size_t index = 0; index < cpu_prefill.value().logits.values.size(); ++index) {
        check_near(
            automatic_prefill.value().logits.values[index],
            cpu_prefill.value().logits.values[index],
            1e-4f);
    }

    RuntimeOptions hybrid_options;
    hybrid_options.hybrid_mode = HybridMode::HybridExperts;
    auto hybrid_model = runtime.load_model(package.path(), hybrid_options);
    if (has_flag(
            runtime.capabilities().flags,
            RuntimeCapabilityVulkanCpuMix)) {
        check(static_cast<bool>(hybrid_model));
        check(static_cast<bool>(hybrid_model.value()->hybrid_mode() == HybridMode::HybridExperts));

        AttentionPackage attention_package;
        auto attention_model = runtime.load_model(attention_package.path(), hybrid_options);
        check(static_cast<bool>(attention_model));
        auto attention_session = runtime.create_session(attention_model.value());
        check(static_cast<bool>(attention_session));
        const std::vector<int32_t> attention_prompt = {0, 1, 0, 1};
        auto attention_prefill = attention_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(attention_prefill));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_attention_blocks == 1));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_linear_dispatches == 3));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_compute_submissions == 2));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_batch_uploads == 2));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_batch_downloads == 2));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_auxiliary_uploads == 3));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_auxiliary_upload_bytes > 0));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_staging_slot_resizes
                                    + attention_session.value()->statistics().vulkan_staging_slot_reuses
                                == 7));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_staging_slot_reuses > 0));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_staging_slot_acquisitions == 2));
        check(static_cast<bool>(attention_session.value()->statistics().kv_cache_logical_bytes > 0));
        check(static_cast<bool>(attention_session.value()->statistics().kv_cache_allocated_bytes
                                >= attention_session.value()->statistics().kv_cache_logical_bytes));

        auto cpu_attention_model = runtime.load_model(attention_package.path(), cpu_options);
        check(static_cast<bool>(cpu_attention_model));
        auto cpu_attention_session = runtime.create_session(cpu_attention_model.value());
        check(static_cast<bool>(cpu_attention_session));
        auto cpu_attention_prefill = cpu_attention_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(cpu_attention_prefill));
        check(static_cast<bool>(cpu_attention_prefill.value().logits.values.size()
                                == attention_prefill.value().logits.values.size()));
        for (size_t index = 0; index < cpu_attention_prefill.value().logits.values.size(); ++index) {
            check_near(
                attention_prefill.value().logits.values[index],
                cpu_attention_prefill.value().logits.values[index],
                1e-4f);
        }
        auto attention_decode = attention_session.value()->decode(1);
        auto cpu_attention_decode = cpu_attention_session.value()->decode(1);
        check(static_cast<bool>(attention_decode));
        check(static_cast<bool>(cpu_attention_decode));
        for (size_t index = 0; index < cpu_attention_decode.value().logits.values.size(); ++index) {
            check_near(
                attention_decode.value().logits.values[index],
                cpu_attention_decode.value().logits.values[index],
                1e-4f);
        }

        AttentionPackage full_attention_package(false, 0);
        auto full_attention_model = runtime.load_model(full_attention_package.path(), hybrid_options);
        check(static_cast<bool>(full_attention_model));
        SessionOptions chunked_attention_options;
        chunked_attention_options.prefill_chunk_size = 2;
        auto chunked_attention_session = runtime.create_session(
            full_attention_model.value(), chunked_attention_options);
        check(static_cast<bool>(chunked_attention_session));
        auto chunked_attention_prefill = chunked_attention_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(chunked_attention_prefill));
        check(static_cast<bool>(chunked_attention_session.value()->statistics().kv_cache_allocated_bytes
                                > chunked_attention_session.value()->statistics().kv_cache_logical_bytes));

        auto full_cpu_model = runtime.load_model(full_attention_package.path(), cpu_options);
        check(static_cast<bool>(full_cpu_model));
        auto full_cpu_session = runtime.create_session(full_cpu_model.value());
        check(static_cast<bool>(full_cpu_session));
        auto full_cpu_prefill = full_cpu_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(full_cpu_prefill));
        for (size_t index = 0; index < full_cpu_prefill.value().logits.values.size(); ++index) {
            check_near(
                chunked_attention_prefill.value().logits.values[index],
                full_cpu_prefill.value().logits.values[index],
                1e-4f);
        }
        auto chunked_attention_decode = chunked_attention_session.value()->decode(1);
        auto full_cpu_decode = full_cpu_session.value()->decode(1);
        check(static_cast<bool>(chunked_attention_decode));
        check(static_cast<bool>(full_cpu_decode));
        for (size_t index = 0; index < full_cpu_decode.value().logits.values.size(); ++index) {
            check_near(
                chunked_attention_decode.value().logits.values[index],
                full_cpu_decode.value().logits.values[index],
                1e-4f);
        }

        RuntimeOptions prefetch_options;
        prefetch_options.hybrid_mode = HybridMode::VulkanWithCpuPrefetch;
        auto prefetch_model = runtime.load_model(package.path(), prefetch_options);
        check(static_cast<bool>(prefetch_model));
        check(static_cast<bool>(prefetch_model.value()->hybrid_mode() == HybridMode::VulkanWithCpuPrefetch));
        auto prefetch_session = runtime.create_session(prefetch_model.value());
        check(static_cast<bool>(prefetch_session));
        check(static_cast<bool>(prefetch_session.value()->prefill(packed_prompt)));
        check(static_cast<bool>(prefetch_session.value()->statistics().expert_prefetches > 0));
        check(static_cast<bool>(prefetch_session.value()->statistics().expert_prefetch_bytes > 0));
    }
    else {
        check(static_cast<bool>(!hybrid_model));
        check(static_cast<bool>(hybrid_model.error().code == ErrorCode::UnsupportedModel));

        RuntimeOptions prefetch_options;
        prefetch_options.hybrid_mode = HybridMode::VulkanWithCpuPrefetch;
        auto prefetch_model = runtime.load_model(package.path(), prefetch_options);
        check(static_cast<bool>(!prefetch_model));
        check(static_cast<bool>(prefetch_model.error().code == ErrorCode::UnsupportedModel));
    }
}

void test_phase_zero_rejects_unimplemented_output_mode()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));

    SessionOptions options;
    options.logits_output_mode = LogitsOutputMode::TopKCandidates;
    auto session = runtime.create_session(model.value(), options);
    check(static_cast<bool>(!session));
    check(static_cast<bool>(session.error().code == ErrorCode::UnsupportedModel));
}

void test_flag_defaults()
{
    RuntimeOptions runtime;
    check(static_cast<bool>(!has_flag(runtime.flags, RuntimeOptionMemoryMapExperts)));

    RuntimeCapabilities capabilities;
    check(static_cast<bool>(has_flag(capabilities.flags, RuntimeCapabilityCpuExecution)));
    check(static_cast<bool>(has_flag(capabilities.flags, RuntimeCapabilityMxfp4CpuKernel)));
    check(static_cast<bool>(has_flag(
        capabilities.flags,
        RuntimeCapabilityCrossSessionScheduling)));

    SchedulerOptions scheduler;
    check(static_cast<bool>(!has_flag(scheduler.flags, SchedulerOptionPinWorkers)));

    ExpertDispatchOptions dispatch;
    check(static_cast<bool>(has_flag(
        dispatch.flags,
        ExpertDispatchNormalizeTopKWeights)));

    MoeDescriptor moe;
    check(static_cast<bool>(has_flag(moe.flags, MoeDescriptorNormalizeTopKWeights)));

    LayerDescriptor layer;
    check(static_cast<bool>(has_flag(layer.flags, LayerDescriptorMoe)));
    check(static_cast<bool>(!has_flag(layer.flags, LayerDescriptorAttention)));

    ExpertPlan expert;
    check(static_cast<bool>(has_flag(expert.flags, ExpertPlanGated)));

    ModelMemoryPlan memory;
    check(static_cast<bool>(!has_flag(memory.flags, ModelMemoryFileBackedExperts)));
}

} // namespace moe
} // namespace ncnn

int main()
{
    try {
        ncnn::moe::test_flag_defaults();
        ncnn::moe::test_ncnn_linear_operator();
        ncnn::moe::test_mxfp4_cpu_kernel_and_fused_gate_up();
        ncnn::moe::test_mapped_file_range_and_shared_buffer();
        ncnn::moe::test_safetensors_dense_mmap();
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
        ncnn::moe::test_moe_ir_execution_graph_and_scheduler();
        ncnn::moe::test_expert_dispatcher_groups_routes();
        ncnn::moe::test_automatic_expert_memory_planning();
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
