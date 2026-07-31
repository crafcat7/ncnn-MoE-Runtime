#include "ncnn/moe/runtime.h"

#include "compiler/moe_ir.hpp"
#include "kernels/cpu_attention.h"
#include "kernels/cpu_gated_delta_net.h"
#include "kernels/cpu_mxfp4.h"
#include "kernels/cpu_ops.h"
#include "kernels/cpu_state_cache.h"
#include "kernels/cpu_float8.h"
#include "kernels/cpu_vector.h"
#include "kernels/cpu_hyper_connection.h"
#include "engine/cpu_executor.h"
#include "engine/cpu_task_worker.h"
#include "engine/expert_backend.h"
#include "engine/cpu_session_state.h"
#include "engine/cpu_topology.h"
#include "storage/expert_cache.h"
#include "fixture_model_adapter.h"
#include "storage/mapped_file.h"
#include "graph/memory_planner.h"
#include "backends/ncnn/ncnn_linear.h"
#include "ncnn/moe/expert_dispatcher.h"
#include "models/builtin_model_adapter.h"
#include "models/deepseek_v4_model_adapter.h"
#include "models/qwen3_5_moe_model_adapter.h"
#include "models/safetensors.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <regex>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

static void check(bool condition, const std::source_location location = std::source_location::current())
{
    if (!condition)
    {
        throw std::runtime_error("test check failed at " + std::string(location.file_name()) + ":" + std::to_string(location.line()));
    }
}

static void check_near(float actual, float expected, float tolerance)
{
    if (std::abs(actual - expected) > tolerance)
    {
        throw std::runtime_error("near check failed: actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
    }
}

class ScopedEnvironmentOverride
{
public:
    ScopedEnvironmentOverride(const char* name, const char* value)
        : name_(name)
    {
#if defined(_WIN32)
        size_t required = 0;
        getenv_s(&required, nullptr, 0, name);
        if (required > 0)
        {
            previous_.resize(required);
            getenv_s(&required, previous_.data(), previous_.size(), name);
            had_previous_ = true;
        }
        _putenv_s(name, value);
#else
        const char* previous = std::getenv(name);
        if (previous)
        {
            previous_ = previous;
            had_previous_ = true;
        }
        setenv(name, value, 1);
#endif
    }

    ~ScopedEnvironmentOverride()
    {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), had_previous_ ? previous_.c_str() : "");
#else
        if (had_previous_)
        {
            setenv(name_.c_str(), previous_.c_str(), 1);
        }
        else
        {
            unsetenv(name_.c_str());
        }
#endif
    }

private:
    std::string name_;
    std::string previous_;
    bool had_previous_ = false;
};

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
    for (uint32_t attempt = 0; attempt < 1000; ++attempt)
    {
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const std::filesystem::path candidate = std::filesystem::temp_directory_path() / (std::string(prefix) + std::to_string(stamp) + "_" + std::to_string(attempt));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error))
            return candidate;
        if (error && error != std::make_error_code(std::errc::file_exists))
            throw std::runtime_error("failed to create temporary test directory: " + error.message());
    }
    throw std::runtime_error("failed to allocate a unique temporary test directory");
}

class ScopedTestDirectory
{
public:
    explicit ScopedTestDirectory(const char* prefix)
        : path_(create_unique_test_directory(prefix))
    {
    }

    ~ScopedTestDirectory()
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
                      "model_type" : "test_moe",
            "vocabulary_size" : 4,
            "hidden_size" : 2,
            "intermediate_size" : 2,
            "layer_count" : 1,
            "expert_count" : 2,
            "experts_per_token" : 1,
            "expert_activation" : "relu",
                                  "expert_layout" : "up_down",
                                                    "normalize_topk_weights" : true,
                                                                               "use_expert_bias" : false,
                                                                                                   "norm_epsilon" : 0.00001,
                                                                                                   "weights_file" : "model.test.bin"
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
        weights.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
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
                      "model_type" : "test_moe",
            "vocabulary_size" : 2,
            "hidden_size" : 2,
            "intermediate_size" : 2,
            "layer_count" : 1,
            "expert_count" : 3,
            "experts_per_token" : 2,
            "expert_activation" : "relu",
                                  "expert_layout" : "up_down",
                                                    "normalize_topk_weights" : true,
                                                                               "norm_epsilon" : 0.00001
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
        weights.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
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
                      "model_type" : "test_moe",
            "vocabulary_size" : 2,
            "hidden_size" : 2,
            "intermediate_size" : 2,
            "layer_count" : 1,
            "expert_count" : 1,
            "experts_per_token" : 1,
            "expert_activation" : "relu",
                                  "expert_layout" : "up_down",
                                                    "expert_weight_dtype" : "int8",
                                                                            "normalize_topk_weights" : true,
                                                                                                       "norm_epsilon" : 0.00001
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
        stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    static void write_int8_matrix(std::ofstream& stream, const std::vector<int8_t>& values, const std::vector<float>& scales)
    {
        stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()));
        write_floats(stream, scales);
    }

    std::filesystem::path path_;
};

class AttentionPackage
{
public:
    explicit AttentionPackage(bool bfloat16_kv_cache = false, uint32_t sliding_window = 2, bool attention_bias = true, bool attention_sinks = true)
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
        auto append = [&values](std::initializer_list<float> additions) { values.insert(values.end(), additions.begin(), additions.end()); };
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
            append({0.375f});
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
        weights.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
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
    check(static_cast<bool>(has_flag(runtime.capabilities().flags, RuntimeCapabilityVulkanAttention) == has_flag(runtime.capabilities().flags, RuntimeCapabilityVulkanCpuMix)));
    check(static_cast<bool>(runtime.capabilities().cpu_linear_thread_limit >= 1));
    check(static_cast<bool>(runtime.capabilities().float8_linear_thread_limit >= 1));
    check(static_cast<bool>(runtime.capabilities().float8_linear_row_group_size == 1 || runtime.capabilities().float8_linear_row_group_size == 2
                            || runtime.capabilities().float8_linear_row_group_size == 4));
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
    const float final_value = normalized_equal / std::sqrt(normalized_equal * normalized_equal + 1e-5f);
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
    const float final_expert_one = expert_one_value / std::sqrt(expert_one_value * expert_one_value / 2.0f + 1e-5f);
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
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        for (uint32_t column = 0; column < output.columns(); ++column)
        {
            float expected = bias.float32_data[column];
            for (uint32_t input_column = 0; input_column < input.columns(); ++input_column)
            {
                expected += matrix.float32_data[column * input.columns() + input_column] * input.row(row_index)[input_column];
            }
            check_near(output.row(row_index)[column], expected, 1e-5f);
        }
    }
    const CpuBatch expected_output = output;
    std::atomic<bool> concurrent_linear_valid{true};
    std::vector<std::thread> linear_workers;
    for (uint32_t worker = 0; worker < 4; ++worker)
    {
        linear_workers.emplace_back([&]() {
            for (uint32_t iteration = 0;
                 iteration < 64;
                 ++iteration)
            {
                CpuBatch concurrent_output;
                if (!linear->forward(input, concurrent_output)
                    || concurrent_output.rows()
                           != expected_output.rows()
                    || concurrent_output.columns()
                           != expected_output.columns())
                {
                    concurrent_linear_valid.store(
                        false,
                        std::memory_order_relaxed);
                    return;
                }
                for (size_t row = 0;
                     row < concurrent_output.rows();
                     ++row)
                {
                    for (uint32_t column = 0;
                         column < concurrent_output.columns();
                         ++column)
                    {
                        if (!std::isfinite(
                                concurrent_output.row(row)[column])
                            || std::abs(
                                   concurrent_output.row(row)[column]
                                   - expected_output.row(row)[column])
                                   > 1e-5f)
                        {
                            concurrent_linear_valid.store(
                                false,
                                std::memory_order_relaxed);
                            return;
                        }
                    }
                }
            }
        });
    }
    for (std::thread& worker : linear_workers)
        worker.join();
    check(concurrent_linear_valid.load(std::memory_order_relaxed));

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
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        for (uint32_t column = 0; column < output.columns(); ++column)
        {
            float expected = bfloat16_to_float(bfloat_bias.bfloat16_data[column]);
            for (uint32_t input_column = 0; input_column < input.columns(); ++input_column)
            {
                expected += bfloat16_to_float(bfloat_matrix.bfloat16_data[column * input.columns() + input_column]) * input.row(row_index)[input_column];
            }
            check_near(output.row(row_index)[column], expected, 1e-5f);
        }
    }

    if (NcnnLinearOperator::vulkan_device_count() > 0)
    {
        const auto vulkan_linear = NcnnLinearOperator::create(matrix, &bias, NcnnLinearDevice::Vulkan);
        check(static_cast<bool>(vulkan_linear));
        const NcnnVulkanRuntimeCounters initial_counters = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
        for (uint32_t iteration = 0; iteration < 4; ++iteration)
        {
            check(static_cast<bool>(vulkan_linear->forward(input, output)));
            for (size_t row_index = 0; row_index < input.rows(); ++row_index)
            {
                for (uint32_t column = 0; column < output.columns(); ++column)
                {
                    float expected = bias.float32_data[column];
                    for (uint32_t input_column = 0; input_column < input.columns(); ++input_column)
                    {
                        expected += matrix.float32_data[column * input.columns() + input_column] * input.row(row_index)[input_column];
                    }
                    check_near(output.row(row_index)[column], expected, 1e-4f);
                }
            }
        }
        const NcnnVulkanRuntimeCounters final_counters = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
        check(static_cast<bool>(final_counters.compute_submissions - initial_counters.compute_submissions == 4));
        check(static_cast<bool>(final_counters.batch_uploads - initial_counters.batch_uploads == 4));
        check(static_cast<bool>(final_counters.batch_downloads - initial_counters.batch_downloads == 4));
        check(static_cast<bool>(final_counters.auxiliary_uploads - initial_counters.auxiliary_uploads == 0));
        check(static_cast<bool>(final_counters.staging_slot_resizes - initial_counters.staging_slot_resizes + final_counters.staging_slot_reuses - initial_counters.staging_slot_reuses == 8));
        check(static_cast<bool>(final_counters.staging_slot_reuses - initial_counters.staging_slot_reuses >= 4));
        check(static_cast<bool>(final_counters.staging_slot_acquisitions - initial_counters.staging_slot_acquisitions == 4));
    }
#endif
}

void test_ncnn_vulkan_float8_operator()
{
#if NCNN_MOE_WITH_NCNN
    if (NcnnLinearOperator::vulkan_device_count() == 0)
        return;

    TensorData matrix;
    matrix.dtype = DType::Float8E4M3;
    matrix.shape = {256, 128};
    const size_t element_count = static_cast<size_t>(matrix.shape[0]) * matrix.shape[1];
    std::shared_ptr<uint8_t[]> storage(new uint8_t[element_count], std::default_delete<uint8_t[]>());
    for (size_t index = 0; index < element_count; ++index)
    {
        const float value = static_cast<float>(static_cast<int>(index % 31) - 15) * 0.03125f;
        storage[index] = float_to_float8_e4m3(value);
    }
    matrix.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
    matrix.mapped_byte_count = element_count;
    matrix.quantization_scales = {0.5f, 2.0f};

    CpuBatch input(2, 128);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
        {
            input.row(row)[column] = static_cast<float>(static_cast<int>((row * input.columns() + column) % 37) - 18) * 0.015625f;
        }
    }
    const CpuBatch cpu_output = linear_batch(matrix, input);
    const auto vulkan = NcnnVulkanFloat8Operator::create(matrix);
    check(static_cast<bool>(vulkan));
    CpuBatch vulkan_output;
    check(static_cast<bool>(vulkan->forward(input, vulkan_output)));
    check(static_cast<bool>(vulkan_output.rows() == cpu_output.rows()));
    check(static_cast<bool>(vulkan_output.columns() == cpu_output.columns()));
    for (size_t row = 0; row < cpu_output.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_output.columns(); ++column)
            check_near(vulkan_output.row(row)[column], cpu_output.row(row)[column], 1e-4f);
    }

    TensorData second_matrix;
    second_matrix.dtype = DType::Float8E4M3;
    second_matrix.shape = {128, 256};
    const size_t second_element_count = static_cast<size_t>(second_matrix.shape[0]) * second_matrix.shape[1];
    std::shared_ptr<uint8_t[]> second_storage(new uint8_t[second_element_count], std::default_delete<uint8_t[]>());
    for (size_t index = 0; index < second_element_count; ++index)
    {
        const float value = static_cast<float>(static_cast<int>((index * 5) % 29) - 14) * 0.015625f;
        second_storage[index] = float_to_float8_e4m3(value);
    }
    second_matrix.mapped_data = std::shared_ptr<const uint8_t>(second_storage, second_storage.get());
    second_matrix.mapped_byte_count = second_element_count;
    second_matrix.quantization_scales = {0.5f, 1.5f};
    const CpuBatch cpu_chain = linear_batch(second_matrix, cpu_output);
    const auto second_vulkan = NcnnVulkanFloat8Operator::create(second_matrix);
    check(static_cast<bool>(second_vulkan));
    CpuBatch vulkan_chain;
    check(static_cast<bool>(vulkan->forward_chain(input, *second_vulkan, vulkan_chain)));
    check(static_cast<bool>(vulkan_chain.rows() == cpu_chain.rows()));
    check(static_cast<bool>(vulkan_chain.columns() == cpu_chain.columns()));
    for (size_t row = 0; row < cpu_chain.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_chain.columns(); ++column)
            check_near(vulkan_chain.row(row)[column], cpu_chain.row(row)[column], 1e-3f);
    }
    TensorData norm_weight;
    norm_weight.dtype = DType::Float32;
    norm_weight.shape = {256};
    norm_weight.float32_data.resize(256);
    for (uint32_t column = 0; column < 256; ++column)
        norm_weight.float32_data[column] = 0.75f + static_cast<float>(column % 9) * 0.03125f;
    constexpr float norm_epsilon = 1e-6f;
    const CpuBatch normalized_cpu_output = rms_norm_batch(cpu_output, norm_weight, norm_epsilon);
    const CpuBatch cpu_norm_chain = linear_batch(second_matrix, normalized_cpu_output);
    check(static_cast<bool>(vulkan->prepare_rms_norm(norm_weight, norm_epsilon)));
    CpuBatch vulkan_norm_chain;
    check(static_cast<bool>(vulkan->forward_rms_norm_chain(input, *second_vulkan, vulkan_norm_chain)));
    for (size_t row = 0; row < cpu_norm_chain.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_norm_chain.columns(); ++column)
            check_near(vulkan_norm_chain.row(row)[column], cpu_norm_chain.row(row)[column], 2e-3f);
    }
    CpuBatch vulkan_parallel_chain;
    CpuBatch vulkan_parallel_output;
    check(static_cast<bool>(vulkan->forward_rms_norm_chain_parallel(
        input,
        *second_vulkan,
        *vulkan,
        vulkan_parallel_chain,
        vulkan_parallel_output)));
    for (size_t row = 0; row < cpu_norm_chain.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_norm_chain.columns(); ++column)
            check_near(vulkan_parallel_chain.row(row)[column], cpu_norm_chain.row(row)[column], 2e-3f);
        for (uint32_t column = 0; column < cpu_output.columns(); ++column)
            check_near(vulkan_parallel_output.row(row)[column], cpu_output.row(row)[column], 1e-4f);
    }
    CpuBatch cpu_activated = cpu_output;
    for (size_t row = 0; row < cpu_activated.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_activated.columns(); ++column)
        {
            const float gate = cpu_output.row(row)[column];
            cpu_activated.row(row)[column] = gate / (1.0f + std::exp(-gate)) * gate;
        }
    }
    const CpuBatch cpu_swiglu_chain = linear_batch(second_matrix, cpu_activated);
    CpuBatch vulkan_swiglu_chain;
    check(static_cast<bool>(vulkan->forward_swiglu_chain(
        input,
        *vulkan,
        *second_vulkan,
        ExpertActivation::DeepSeekSwiGlu,
        0.0f,
        vulkan_swiglu_chain)));
    for (size_t row = 0; row < cpu_swiglu_chain.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_swiglu_chain.columns(); ++column)
            check_near(vulkan_swiglu_chain.row(row)[column], cpu_swiglu_chain.row(row)[column], 2e-3f);
    }
#endif
}

void test_ncnn_vulkan_bfloat16_operator()
{
#if NCNN_MOE_WITH_NCNN
    if (NcnnLinearOperator::vulkan_device_count() == 0)
        return;

    TensorData first;
    first.dtype = DType::BFloat16;
    first.shape = {192, 128};
    first.bfloat16_data.resize(first.element_count());
    for (size_t index = 0; index < first.element_count(); ++index)
    {
        const float value =
            static_cast<float>(
                static_cast<int>((index * 17) % 97) - 48)
            * 0.0013f;
        first.bfloat16_data[index] = float_to_bfloat16(value);
    }
    TensorData second;
    second.dtype = DType::BFloat16;
    second.shape = {64, 128};
    second.bfloat16_data.resize(second.element_count());
    for (size_t index = 0; index < second.element_count(); ++index)
    {
        const float value =
            static_cast<float>(
                static_cast<int>((index * 11) % 71) - 35)
            * 0.0017f;
        second.bfloat16_data[index] = float_to_bfloat16(value);
    }
    TensorData first_bias;
    first_bias.dtype = DType::BFloat16;
    first_bias.shape = {192};
    first_bias.bfloat16_data.resize(192);
    for (uint32_t index = 0; index < 192; ++index)
    {
        first_bias.bfloat16_data[index] =
            float_to_bfloat16(
                static_cast<float>(static_cast<int>(index % 13) - 6)
                * 0.0021f);
    }
    TensorData second_bias;
    second_bias.dtype = DType::Float32;
    second_bias.shape = {64};
    second_bias.float32_data.resize(64);
    for (uint32_t index = 0; index < 64; ++index)
    {
        second_bias.float32_data[index] =
            static_cast<float>(static_cast<int>(index % 9) - 4)
            * 0.0019f;
    }

    CpuBatch input(3, 128);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
        {
            input.row(row)[column] =
                static_cast<float>(
                    static_cast<int>(
                        (row * input.columns() + column * 7) % 89)
                    - 44)
                * 0.0031f;
        }
    }
    const CpuBatch first_cpu =
        linear_batch(first, first_bias, input);
    const auto first_vulkan =
        NcnnVulkanBfloat16Operator::create(
            first,
            &first_bias);
    check(static_cast<bool>(first_vulkan));
    CpuBatch first_output;
    check(static_cast<bool>(
        first_vulkan->forward(input, first_output)));
    for (size_t row = 0; row < first_cpu.rows(); ++row)
    {
        for (uint32_t column = 0;
             column < first_cpu.columns();
             ++column)
        {
            check_near(
                first_output.row(row)[column],
                first_cpu.row(row)[column],
                2e-4f);
        }
    }

    CpuBatch one_row(1, input.columns());
    std::copy_n(
        input.row(1),
        input.columns(),
        one_row.row(0));
    CpuBatch one_row_output;
    check(static_cast<bool>(
        first_vulkan->forward(one_row, one_row_output)));
    for (uint32_t column = 0;
         column < first_output.columns();
         ++column)
    {
        check_near(
            one_row_output.row(0)[column],
            first_output.row(1)[column],
            1e-6f);
    }

    const auto fused =
        NcnnVulkanBfloat16Operator::create_fused(
            {&first, &second},
            {&first_bias, &second_bias});
    check(static_cast<bool>(fused));
    CpuBatch fused_output;
    check(static_cast<bool>(
        fused->forward(input, fused_output)));
    const CpuBatch second_cpu =
        linear_batch(second, second_bias, input);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < 192; ++column)
        {
            check_near(
                fused_output.row(row)[column],
                first_cpu.row(row)[column],
                2e-4f);
        }
        for (uint32_t column = 0; column < 64; ++column)
        {
            check_near(
                fused_output.row(row)[192 + column],
                second_cpu.row(row)[column],
                2e-4f);
        }
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
    for (size_t row = 0; row < 4; ++row)
    {
        for (size_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t low = static_cast<uint8_t>((byte + row) % 16);
            const uint8_t high = static_cast<uint8_t>((byte * 3 + row + 1) % 16);
            matrix.mxfp4_blocks[row * 16 + byte] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    CpuBatch input(2, 32);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>(column % 7) - 3) * (row == 0 ? 0.25f : -0.125f);
    }
    static constexpr float values[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    auto scalar_row = [&](size_t matrix_row, size_t input_row) {
        float sum = 0.0f;
        for (size_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t packed = matrix.mxfp4_blocks[matrix_row * 16 + byte];
            sum += values[packed & 0x0f] * input.row(input_row)[byte * 2];
            sum += values[packed >> 4] * input.row(input_row)[byte * 2 + 1];
        }
        return sum;
    };

    const CpuBatch projected = linear_batch(matrix, input);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
    {
        for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
        {
            check_near(projected.row(input_row)[matrix_row], scalar_row(matrix_row, input_row), 1e-5f);
        }
    }
    auto vulkan_projection = NcnnVulkanMxfp4Operator::create(matrix);
    if (NcnnLinearOperator::vulkan_device_count() > 0)
    {
        check(static_cast<bool>(vulkan_projection));
        check(static_cast<bool>(vulkan_projection->input_columns() == 32));
        check(static_cast<bool>(vulkan_projection->output_columns() == 4));
        CpuBatch vulkan_output;
        check(static_cast<bool>(vulkan_projection->forward(input, vulkan_output)));
        for (size_t input_row = 0; input_row < input.rows(); ++input_row)
        {
            for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
            {
                check_near(vulkan_output.row(input_row)[matrix_row], projected.row(input_row)[matrix_row], 1e-4f);
            }
        }
        CpuBatch four_row_input(4, 32);
        for (size_t row = 0; row < four_row_input.rows(); ++row)
        {
            for (uint32_t column = 0; column < four_row_input.columns(); ++column)
            {
                four_row_input.row(row)[column] = static_cast<float>(static_cast<int>((column + row * 5) % 11) - 5) * 0.0625f;
            }
        }
        const CpuBatch four_row_cpu = linear_batch(matrix, four_row_input);
        CpuBatch four_row_vulkan;
        check(static_cast<bool>(vulkan_projection->forward(four_row_input, four_row_vulkan)));
        for (size_t row = 0; row < four_row_input.rows(); ++row)
        {
            for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
            {
                check_near(four_row_vulkan.row(row)[matrix_row], four_row_cpu.row(row)[matrix_row], 1e-4f);
            }
        }
    }
    else
    {
        check(static_cast<bool>(!vulkan_projection));
    }

    CpuBatch decode_input(1, 32);
    std::copy_n(input.row(0), input.columns(), decode_input.row(0));
    const CpuBatch decoded = linear_batch(matrix, decode_input);
    for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
    {
        check_near(decoded.row(0)[matrix_row], scalar_row(matrix_row, 0), 1e-5f);
    }
    TensorData odd_matrix = matrix;
    odd_matrix.shape[0] = 3;
    odd_matrix.mxfp4_blocks.resize(3 * 16);
    odd_matrix.mxfp4_scales.resize(3);
    const CpuBatch odd_projected = linear_batch(odd_matrix, input);
    check(static_cast<bool>(odd_projected.columns() == 3));
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
    {
        for (size_t matrix_row = 0; matrix_row < 3; ++matrix_row)
        {
            check_near(odd_projected.row(input_row)[matrix_row], scalar_row(matrix_row, input_row), 1e-5f);
        }
    }
    auto odd_vulkan_projection = NcnnVulkanMxfp4Operator::create(odd_matrix);
    if (NcnnLinearOperator::vulkan_device_count() > 0)
    {
        check(static_cast<bool>(odd_vulkan_projection));
        CpuBatch odd_vulkan_output;
        check(static_cast<bool>(odd_vulkan_projection->forward(input, odd_vulkan_output)));
        for (size_t input_row = 0; input_row < input.rows(); ++input_row)
        {
            for (size_t matrix_row = 0; matrix_row < 3; ++matrix_row)
            {
                check_near(odd_vulkan_output.row(input_row)[matrix_row], odd_projected.row(input_row)[matrix_row], 1e-4f);
            }
        }
    }
    else
    {
        check(static_cast<bool>(!odd_vulkan_projection));
    }

    TensorData bias;
    bias.dtype = DType::Float32;
    bias.shape = {4};
    bias.float32_data = {0.25f, -0.5f, 0.75f, -1.0f};
    const CpuBatch fused = fused_mxfp4_gate_up_batch(matrix, &bias, input, ExpertActivation::GptOssSwiGlu, 7.0f);
    check(static_cast<bool>(fused.rows() == input.rows()));
    check(static_cast<bool>(fused.columns() == 2));
    for (float sigmoid_scale : {1.0f, 1.702f})
    {
        for (int step = -1000; step <= 700; ++step)
        {
            const float value = static_cast<float>(step) * 0.01f;
            const float expected = value / (1.0f + std::exp(-sigmoid_scale * value));
            check_near(approximate_scaled_silu(value, sigmoid_scale), expected, 1e-5f);
            check_near(scaled_silu(value, sigmoid_scale), expected, 1e-5f);
        }
    }
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
    {
        for (size_t column = 0; column < fused.columns(); ++column)
        {
            const float gate = std::min(scalar_row(column * 2, input_row) + bias.float32_data[column * 2], 7.0f);
            const float linear = std::clamp(scalar_row(column * 2 + 1, input_row) + bias.float32_data[column * 2 + 1], -7.0f, 7.0f);
            const float expected = gate / (1.0f + std::exp(-1.702f * gate)) * (linear + 1.0f);
            check_near(fused.row(input_row)[column], expected, 1e-5f);
        }
    }

    TensorData expert_gate_up;
    expert_gate_up.dtype = DType::MxFp4;
    expert_gate_up.shape = {64, 32};
    expert_gate_up.mxfp4_scales.resize(64);
    expert_gate_up.mxfp4_blocks.resize(64 * 16);
    for (size_t row = 0; row < 64; ++row)
    {
        expert_gate_up.mxfp4_scales[row] = static_cast<uint8_t>(125 + row % 5);
        for (size_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t low = static_cast<uint8_t>((row * 3 + byte * 5 + 1) % 16);
            const uint8_t high = static_cast<uint8_t>((row * 7 + byte * 2 + 4) % 16);
            expert_gate_up.mxfp4_blocks[row * 16 + byte] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    TensorData expert_gate_up_bias;
    expert_gate_up_bias.dtype = DType::Float32;
    expert_gate_up_bias.shape = {64};
    expert_gate_up_bias.float32_data.resize(64);
    for (size_t row = 0; row < 64; ++row)
    {
        expert_gate_up_bias.float32_data[row] = static_cast<float>(static_cast<int>(row % 9) - 4) * 0.03125f;
    }
    TensorData expert_down;
    expert_down.dtype = DType::MxFp4;
    expert_down.shape = {7, 32};
    expert_down.mxfp4_scales.resize(7);
    expert_down.mxfp4_blocks.resize(7 * 16);
    for (size_t row = 0; row < 7; ++row)
    {
        expert_down.mxfp4_scales[row] = static_cast<uint8_t>(126 + row % 3);
        for (size_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t low = static_cast<uint8_t>((row * 11 + byte * 3 + 2) % 16);
            const uint8_t high = static_cast<uint8_t>((row * 5 + byte * 7 + 8) % 16);
            expert_down.mxfp4_blocks[row * 16 + byte] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    TensorData expert_down_bias;
    expert_down_bias.dtype = DType::Float32;
    expert_down_bias.shape = {7};
    expert_down_bias.float32_data.resize(7);
    for (size_t row = 0; row < 7; ++row)
    {
        expert_down_bias.float32_data[row] = static_cast<float>(static_cast<int>(row) - 3) * 0.0625f;
    }
    constexpr float expert_activation_limit = 5.25f;
    CpuBatch repeated_expert_input(4, 32);
    for (uint32_t column = 0;
         column < repeated_expert_input.columns();
         ++column)
    {
        repeated_expert_input.row(0)[column] = static_cast<float>(
                                                   static_cast<int>(column % 11) - 5)
                                               * 0.03125f;
        repeated_expert_input.row(1)[column] = static_cast<float>(
                                                   static_cast<int>((column * 3) % 17) - 8)
                                               * 0.015625f;
        repeated_expert_input.row(3)[column] = static_cast<float>(
                                                   static_cast<int>((column * 5) % 19) - 9)
                                               * 0.015625f;
    }
    std::copy_n(
        repeated_expert_input.row(0),
        repeated_expert_input.columns(),
        repeated_expert_input.row(2));
    const CpuBatch repeated_activated = fused_mxfp4_gate_up_batch(
        expert_gate_up,
        &expert_gate_up_bias,
        repeated_expert_input,
        ExpertActivation::DeepSeekSwiGlu,
        expert_activation_limit);
    const CpuBatch repeated_reference = linear_batch(
        expert_down,
        expert_down_bias,
        repeated_activated);
    CpuBatch repeated_output;
    Mxfp4Task repeated_task;
    repeated_task.gate_up = &expert_gate_up;
    repeated_task.gate_up_bias = &expert_gate_up_bias;
    repeated_task.down = &expert_down;
    repeated_task.down_bias = &expert_down_bias;
    repeated_task.input = &repeated_expert_input;
    repeated_task.output = &repeated_output;
    repeated_task.activation = ExpertActivation::DeepSeekSwiGlu;
    repeated_task.activation_limit = expert_activation_limit;
    Mxfp4Scratch repeated_scratch;
    check(static_cast<bool>(
        mxfp4_expert_batch(
            std::span<const Mxfp4Task>(
                &repeated_task,
                1),
            &repeated_scratch)));
    check(static_cast<bool>(
        repeated_scratch.physical_input_rows
        == std::vector<uint32_t>({3})));
    check(static_cast<bool>(
        repeated_output.rows()
        == repeated_reference.rows()));
    for (size_t row = 0;
         row < repeated_output.rows();
         ++row)
    {
        for (uint32_t column = 0;
             column < repeated_output.columns();
             ++column)
        {
            check_near(
                repeated_output.row(row)[column],
                repeated_reference.row(row)[column],
                1e-5f);
        }
    }
    for (size_t token_count : {size_t(1), repeated_expert_input.rows()})
    {
        CpuBatch silu_input(token_count, repeated_expert_input.columns());
        for (size_t row = 0; row < token_count; ++row)
        {
            std::copy_n(
                repeated_expert_input.row(row),
                repeated_expert_input.columns(),
                silu_input.row(row));
        }
        const CpuBatch silu_activated = fused_mxfp4_gate_up_batch(
            expert_gate_up,
            nullptr,
            silu_input,
            ExpertActivation::Silu,
            0.0f);
        const CpuBatch silu_reference = linear_batch(
            expert_down,
            silu_activated);
        CpuBatch silu_output;
        Mxfp4Task silu_task;
        silu_task.gate_up = &expert_gate_up;
        silu_task.down = &expert_down;
        silu_task.input = &silu_input;
        silu_task.output = &silu_output;
        silu_task.activation = ExpertActivation::Silu;
        Mxfp4Scratch silu_scratch;
        check(static_cast<bool>(
            mxfp4_expert_batch(
                std::span<const Mxfp4Task>(&silu_task, 1),
                &silu_scratch)));
        check(silu_output.rows() == silu_reference.rows());
        check(silu_output.columns() == silu_reference.columns());
        for (size_t row = 0; row < silu_output.rows(); ++row)
        {
            for (uint32_t column = 0; column < silu_output.columns(); ++column)
            {
                check_near(
                    silu_output.row(row)[column],
                    silu_reference.row(row)[column],
                    1e-5f);
            }
        }
    }
    repeated_expert_input.row(2)[0] += 0.03125f;
    check(static_cast<bool>(
        mxfp4_expert_batch(
            std::span<const Mxfp4Task>(
                &repeated_task,
                1),
            &repeated_scratch)));
    check(static_cast<bool>(
        repeated_scratch.physical_input_rows
        == std::vector<uint32_t>({4})));

    auto vulkan_expert = NcnnVulkanMxfp4ExpertOperator::create(expert_gate_up, &expert_gate_up_bias, expert_down, &expert_down_bias, expert_activation_limit);
    if (NcnnLinearOperator::vulkan_device_count() > 0)
    {
        check(static_cast<bool>(vulkan_expert));
        for (size_t token_count : {size_t(1), size_t(2), size_t(4)})
        {
            CpuBatch expert_input(token_count, 32);
            for (size_t row = 0; row < expert_input.rows(); ++row)
            {
                for (uint32_t column = 0; column < expert_input.columns(); ++column)
                {
                    expert_input.row(row)[column] = static_cast<float>(static_cast<int>((column * 5 + row * 7) % 17) - 8) * 0.015625f;
                }
            }
            const CpuBatch cpu_activated = fused_mxfp4_gate_up_batch(expert_gate_up, &expert_gate_up_bias, expert_input, ExpertActivation::GptOssSwiGlu, expert_activation_limit);
            const CpuBatch cpu_expert = linear_batch(expert_down, expert_down_bias, cpu_activated);
            CpuBatch vulkan_expert_output;
            check(static_cast<bool>(vulkan_expert->forward(expert_input, vulkan_expert_output)));
            check(static_cast<bool>(vulkan_expert_output.rows() == cpu_expert.rows()));
            check(static_cast<bool>(vulkan_expert_output.columns() == cpu_expert.columns()));
            for (size_t row = 0; row < cpu_expert.rows(); ++row)
            {
                for (uint32_t column = 0; column < cpu_expert.columns(); ++column)
                {
                    check_near(vulkan_expert_output.row(row)[column], cpu_expert.row(row)[column], 1e-3f);
                }
            }
        }
        auto deepseek_vulkan_expert = NcnnVulkanMxfp4ExpertOperator::create(
            expert_gate_up,
            &expert_gate_up_bias,
            expert_down,
            &expert_down_bias,
            expert_activation_limit,
            automatic_vulkan_device_index,
            ExpertActivation::DeepSeekSwiGlu);
        check(static_cast<bool>(deepseek_vulkan_expert));
        CpuBatch deepseek_input(1, 32);
        for (uint32_t column = 0; column < deepseek_input.columns(); ++column)
            deepseek_input.row(0)[column] = static_cast<float>(static_cast<int>((column * 3) % 19) - 9) * 0.015625f;
        const CpuBatch deepseek_activated = fused_mxfp4_gate_up_batch(
            expert_gate_up,
            &expert_gate_up_bias,
            deepseek_input,
            ExpertActivation::DeepSeekSwiGlu,
            expert_activation_limit);
        const CpuBatch deepseek_expected = linear_batch(expert_down, expert_down_bias, deepseek_activated);
        CpuBatch deepseek_actual;
        check(static_cast<bool>(deepseek_vulkan_expert->forward(deepseek_input, deepseek_actual)));
        for (uint32_t column = 0; column < deepseek_expected.columns(); ++column)
            check_near(deepseek_actual.row(0)[column], deepseek_expected.row(0)[column], 1e-3f);
    }
    else
    {
        check(static_cast<bool>(!vulkan_expert));
    }

    auto expert_backend = create_vulkan_mxfp4_expert_backend(4096);
    if (NcnnLinearOperator::vulkan_device_count() > 0)
    {
        check(static_cast<bool>(expert_backend));
        const auto backend_gate_up = std::make_shared<TensorData>(expert_gate_up);
        const auto backend_down = std::make_shared<TensorData>(expert_down);
        expert_backend->admit("test-expert", backend_gate_up, &expert_gate_up_bias, backend_down, &expert_down_bias, 0, 1, expert_activation_limit);
        expert_backend->admit("test-expert", backend_gate_up, &expert_gate_up_bias, backend_down, &expert_down_bias, 0, 1, expert_activation_limit);
        const auto admission_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (expert_backend->statistics().stores == 0 && std::chrono::steady_clock::now() < admission_deadline)
        {
            std::this_thread::yield();
        }
        check(static_cast<bool>(expert_backend->statistics().stores == 1));
        const uint64_t backend_weight_bytes = expert_gate_up.mxfp4_blocks.size() + expert_gate_up.mxfp4_scales.size() + expert_down.mxfp4_blocks.size() + expert_down.mxfp4_scales.size();
        for (uint32_t sample = 0; sample < 3; ++sample)
        {
            expert_backend->observe_cpu(1, backend_weight_bytes, 1000);
        }
        CpuBatch backend_input(1, 32);
        for (uint32_t column = 0; column < backend_input.columns(); ++column)
        {
            backend_input.row(0)[column] = static_cast<float>(static_cast<int>(column % 13) - 6) * 0.015625f;
        }
        const CpuBatch backend_activated = fused_mxfp4_gate_up_batch(expert_gate_up, &expert_gate_up_bias, backend_input, ExpertActivation::GptOssSwiGlu, expert_activation_limit);
        const CpuBatch backend_expected = linear_batch(expert_down, expert_down_bias, backend_activated);
        for (uint32_t sample = 0; sample < 3; ++sample)
        {
            CpuBatch backend_output;
            check(static_cast<bool>(expert_backend->try_execute("test-expert", backend_input, backend_output) == ExpertBackendExecutionResult ::Executed));
            for (uint32_t column = 0; column < backend_expected.columns(); ++column)
            {
                check_near(backend_output.row(0)[column], backend_expected.row(0)[column], 1e-3f);
            }
        }
        CpuBatch backend_batch_output_first;
        CpuBatch backend_batch_output_second;
        const std::array<ExpertBackendRequest, 2> backend_requests = {{
            {
                "test-expert",
                &backend_input,
                &backend_batch_output_first,
            },
            {
                "test-expert",
                &backend_input,
                &backend_batch_output_second,
            },
        }};
        const auto backend_batch_results = expert_backend->try_execute_batch(backend_requests);
        check(static_cast<bool>(backend_batch_results.size() == 2));
        for (ExpertBackendExecutionResult result : backend_batch_results)
        {
            check(static_cast<bool>(result == ExpertBackendExecutionResult ::Executed));
        }
        for (const CpuBatch* batch_output : {&backend_batch_output_first, &backend_batch_output_second})
        {
            for (uint32_t column = 0; column < backend_expected.columns(); ++column)
            {
                check_near(batch_output->row(0)[column], backend_expected.row(0)[column], 1e-3f);
            }
        }
        const ExpertBackendStatistics backend_statistics = expert_backend->statistics();
        check(static_cast<bool>(backend_statistics.executions == 5));
        check(static_cast<bool>(backend_statistics.hits == 5));
        check(static_cast<bool>(backend_statistics.bytes_uploaded > 0));
        check(static_cast<bool>(backend_statistics.arc_frequent_bytes > 0));

        auto device_source = create_vulkan_victim_cache(4096);
        check(static_cast<bool>(device_source));
        device_source->admit("victim-expert", backend_gate_up, backend_down, 0,
                             {
                                 &expert_gate_up_bias,
                                 &expert_down_bias,
                                 expert_activation_limit,
                                 true,
                             });
        device_source->wait_for_background_work();
        check(static_cast<bool>(device_source->statistics().stores == 1));
        auto source_backend = create_vulkan_mxfp4_expert_backend(0, automatic_vulkan_device_index, device_source);
        check(static_cast<bool>(source_backend));
        CpuBatch source_output;
        check(static_cast<bool>(source_backend->try_execute("victim-expert", backend_input, source_output) == ExpertBackendExecutionResult ::Executed));
        for (uint32_t column = 0; column < backend_expected.columns(); ++column)
        {
            check_near(source_output.row(0)[column], backend_expected.row(0)[column], 1e-3f);
        }
        const ExpertBackendStatistics source_statistics = source_backend->statistics();
        check(static_cast<bool>(source_statistics.device_source_hits == 1));
        check(static_cast<bool>(source_statistics.device_source_executions == 1));
        check(static_cast<bool>(source_statistics.device_source_execution_failures == 0));
        check(static_cast<bool>(device_source->statistics().bytes_downloaded == 0));
    }
    else
    {
        check(static_cast<bool>(!expert_backend));
    }

    constexpr uint32_t test_block_count = 4;
    constexpr size_t test_token_count = 3;
    constexpr size_t test_input_columns = test_block_count * 32;
    constexpr size_t test_input_stride = test_input_columns + 5;
    constexpr size_t test_output_stride = 3;
    std::vector<uint8_t> first_packed(test_block_count * 16);
    std::vector<uint8_t> second_packed(test_block_count * 16);
    for (size_t index = 0; index < first_packed.size(); ++index)
    {
        first_packed[index] = static_cast<uint8_t>(((index * 5 + 3) % 16) | (((index * 7 + 1) % 16) << 4));
        second_packed[index] = static_cast<uint8_t>(((index * 11 + 2) % 16) | (((index * 3 + 9) % 16) << 4));
    }
    const std::array<uint8_t, test_block_count> first_scales = {125, 127, 128, 126};
    const std::array<uint8_t, test_block_count> second_scales = {128, 124, 127, 129};
    std::vector<float> strided_input(test_token_count * test_input_stride);
    for (size_t token = 0; token < test_token_count; ++token)
    {
        for (size_t column = 0; column < test_input_stride; ++column)
        {
            strided_input[token * test_input_stride + column] = static_cast<float>(static_cast<int>((column * 13 + token * 5) % 23) - 11) * 0.03125f;
        }
    }
    auto reference_dot = [&](const std::vector<uint8_t>& packed, const std::array<uint8_t, test_block_count>& scales, size_t token) {
        float sum = 0.0f;
        for (uint32_t block = 0; block < test_block_count; ++block)
        {
            float block_sum = 0.0f;
            for (uint32_t byte = 0; byte < 16; ++byte)
            {
                const uint8_t value = packed[block * 16 + byte];
                const size_t input_offset = token * test_input_stride + block * 32 + byte * 2;
                block_sum += values[value & 0x0f] * strided_input[input_offset];
                block_sum += values[value >> 4] * strided_input[input_offset + 1];
            }
            sum += block_sum * std::ldexp(1.0f, static_cast<int>(scales[block]) - 127);
        }
        return sum;
    };

    check_near(mxfp4_dot(first_packed.data(), first_scales.data(), test_block_count, strided_input.data()), reference_dot(first_packed, first_scales, 0), 1e-4f);

    std::vector<float> gemm_output(test_token_count * test_output_stride, -999.0f);
    mxfp4_gemm_row(first_packed.data(), first_scales.data(), test_block_count, strided_input.data(), test_input_stride, test_token_count, gemm_output.data(), test_output_stride);
    for (size_t token = 0; token < test_token_count; ++token)
    {
        check_near(gemm_output[token * test_output_stride], reference_dot(first_packed, first_scales, token), 1e-4f);
    }

    std::vector<float> paired_first(test_token_count * test_output_stride, -999.0f);
    std::vector<float> paired_second(test_token_count * test_output_stride, -999.0f);
    mxfp4_matmul_rows2(first_packed.data(), first_scales.data(), second_packed.data(), second_scales.data(), test_block_count, strided_input.data(), test_input_stride, test_token_count, paired_first.data(), test_output_stride,
                       paired_second.data(), test_output_stride);
    for (size_t token = 0; token < test_token_count; ++token)
    {
        check_near(paired_first[token * test_output_stride], reference_dot(first_packed, first_scales, token), 1e-4f);
        check_near(paired_second[token * test_output_stride], reference_dot(second_packed, second_scales, token), 1e-4f);
    }

    constexpr uint32_t bulk_pair_count = 2;
    constexpr size_t bulk_output_stride = 7;
    std::vector<uint8_t> bulk_packed(bulk_pair_count * 2 * test_block_count * 16);
    std::vector<uint8_t> bulk_scales(bulk_pair_count * 2 * test_block_count);
    const size_t packed_row_bytes = test_block_count * 16;
    for (uint32_t pair = 0; pair < bulk_pair_count; ++pair)
    {
        const std::vector<uint8_t>& first_source = pair == 0 ? first_packed : second_packed;
        const std::vector<uint8_t>& second_source = pair == 0 ? second_packed : first_packed;
        const auto& first_scale_source = pair == 0 ? first_scales : second_scales;
        const auto& second_scale_source = pair == 0 ? second_scales : first_scales;
        std::copy(first_source.begin(), first_source.end(), bulk_packed.begin() + pair * 2 * packed_row_bytes);
        std::copy(second_source.begin(), second_source.end(), bulk_packed.begin() + (pair * 2 + 1) * packed_row_bytes);
        std::copy(first_scale_source.begin(), first_scale_source.end(), bulk_scales.begin() + pair * 2 * test_block_count);
        std::copy(second_scale_source.begin(), second_scale_source.end(), bulk_scales.begin() + (pair * 2 + 1) * test_block_count);
    }
    std::vector<float> bulk_output(test_token_count * bulk_output_stride, -999.0f);
    mxfp4_matmul_row_pairs(bulk_packed.data(), bulk_scales.data(), test_block_count, bulk_pair_count, strided_input.data(), test_input_stride, test_token_count, bulk_output.data() + 1, 2, bulk_output_stride, bulk_output.data() + 2, 2,
                           bulk_output_stride);
    for (size_t token = 0; token < test_token_count; ++token)
    {
        check_near(bulk_output[token * bulk_output_stride + 1], reference_dot(first_packed, first_scales, token), 1e-4f);
        check_near(bulk_output[token * bulk_output_stride + 2], reference_dot(second_packed, second_scales, token), 1e-4f);
        check_near(bulk_output[token * bulk_output_stride + 3], reference_dot(second_packed, second_scales, token), 1e-4f);
        check_near(bulk_output[token * bulk_output_stride + 4], reference_dot(first_packed, first_scales, token), 1e-4f);
    }

    constexpr size_t paired_token_count = 2;
    std::vector<float> paired_bulk_output(paired_token_count * bulk_output_stride, -999.0f);
    mxfp4_matmul_row_pairs(bulk_packed.data(), bulk_scales.data(), test_block_count, bulk_pair_count, strided_input.data(), test_input_stride, paired_token_count, paired_bulk_output.data() + 1, 2, bulk_output_stride,
                           paired_bulk_output.data() + 2, 2, bulk_output_stride);
    for (size_t token = 0; token < paired_token_count; ++token)
    {
        check_near(paired_bulk_output[token * bulk_output_stride + 1], reference_dot(first_packed, first_scales, token), 1e-4f);
        check_near(paired_bulk_output[token * bulk_output_stride + 2], reference_dot(second_packed, second_scales, token), 1e-4f);
        check_near(paired_bulk_output[token * bulk_output_stride + 3], reference_dot(second_packed, second_scales, token), 1e-4f);
        check_near(paired_bulk_output[token * bulk_output_stride + 4], reference_dot(first_packed, first_scales, token), 1e-4f);
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    check(static_cast<bool>(mxfp4_kernel_kind() == MxFp4KernelKind::ArmNeon || mxfp4_kernel_kind() == MxFp4KernelKind::ArmSve2));
#endif
    check(static_cast<bool>(std::string(mxfp4_kernel_name()).size() > 0));
    check(static_cast<bool>(std::string(scaled_silu_kernel_name()).size() > 0));
}

class TestExpertVictimCache final : public IExpertVictimCache
{
public:
    explicit TestExpertVictimCache(uint64_t capacity_bytes)
        : capacity_bytes_(capacity_bytes)
    {
    }

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, std::shared_ptr<const TensorData> down, uint32_t, ExpertVictimExecutionMetadata) override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        ExpertVictimPair pair;
        pair.gate_up = std::make_shared<TensorData>(*gate_up);
        pair.down = std::make_shared<TensorData>(*down);
        entries_[std::move(key)] = std::move(pair);
        ++statistics_.admissions;
        ++statistics_.stores;
        statistics_.resident_bytes += gate_up->mxfp4_blocks.size() + gate_up->mxfp4_scales.size() + down->mxfp4_blocks.size() + down->mxfp4_scales.size();
    }

    std::optional<ExpertVictimPair> restore(const std::string& key, const TensorData&, const TensorData&) override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        const auto existing = entries_.find(key);
        if (existing == entries_.end())
        {
            ++statistics_.misses;
            return std::nullopt;
        }
        ++statistics_.hits;
        return existing->second;
    }

    void wait_for_background_work() override
    {
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

class DelayedExpertVictimCache final : public IExpertVictimCache
{
public:
    void admit(std::string, std::shared_ptr<const TensorData>, std::shared_ptr<const TensorData>, uint32_t, ExpertVictimExecutionMetadata) override
    {
    }

    std::optional<ExpertVictimPair> restore(const std::string& key, const TensorData& gate_up_source, const TensorData& down_source) override
    {
        std::this_thread::sleep_for(key == "slow" ? std::chrono::milliseconds(100) : std::chrono::milliseconds(1));
        ExpertVictimPair pair;
        pair.gate_up = make_tensor(gate_up_source, 7);
        pair.down = make_tensor(down_source, 9);
        return pair;
    }

    void wait_for_background_work() override
    {
    }

    ExpertVictimCacheStatistics statistics() const override
    {
        return {};
    }

    uint64_t capacity_bytes() const noexcept override
    {
        return 68;
    }

private:
    static std::shared_ptr<TensorData> make_tensor(const TensorData& source, uint8_t value)
    {
        auto result = std::make_shared<TensorData>(source);
        result->mxfp4_file_storage.reset();
        result->mxfp4_blocks.resize(source.mxfp4_file_storage->blocks_bytes);
        std::fill_n(result->mxfp4_blocks.data(), result->mxfp4_blocks.size(), value);
        result->mxfp4_scales.resize(source.mxfp4_file_storage->scales_bytes);
        std::fill_n(result->mxfp4_scales.data(), result->mxfp4_scales.size(), uint8_t{127});
        return result;
    }
};

void test_sharded_expert_victim_cache()
{
    auto first = std::make_shared<TestExpertVictimCache>(34);
    auto second = std::make_shared<TestExpertVictimCache>(68);
    auto sharded = create_sharded_victim_cache({
        first,
        nullptr,
        second,
    });
    check(static_cast<bool>(sharded));
    check(static_cast<bool>(sharded->capacity_bytes() == 102));

    auto gate_up = std::make_shared<TensorData>();
    gate_up->dtype = DType::MxFp4;
    gate_up->shape = {2, 32};
    gate_up->mxfp4_blocks.resize(32);
    std::fill_n(gate_up->mxfp4_blocks.data(), gate_up->mxfp4_blocks.size(), uint8_t{7});
    gate_up->mxfp4_scales.resize(2);
    std::fill_n(gate_up->mxfp4_scales.data(), gate_up->mxfp4_scales.size(), uint8_t{127});
    auto down = std::make_shared<TensorData>();
    down->dtype = DType::MxFp4;
    down->shape = {1, 1};
    down->mxfp4_blocks.resize(16);
    std::fill_n(down->mxfp4_blocks.data(), down->mxfp4_blocks.size(), uint8_t{9});
    down->mxfp4_scales.resize(1);
    std::fill_n(down->mxfp4_scales.data(), down->mxfp4_scales.size(), uint8_t{126});

    sharded->admit("layer.0.expert.1", gate_up, down);
    sharded->admit("layer.7.expert.3", gate_up, down);
    sharded->wait_for_background_work();
    auto restored_first = sharded->restore("layer.0.expert.1", *gate_up, *down);
    auto restored_second = sharded->restore("layer.7.expert.3", *gate_up, *down);
    check(static_cast<bool>(restored_first));
    check(static_cast<bool>(restored_second));
    check(static_cast<bool>(restored_first->gate_up->mxfp4_blocks.front() == 7));
    check(static_cast<bool>(restored_second->down->mxfp4_blocks.front() == 9));
    const ExpertVictimCacheStatistics statistics = sharded->statistics();
    check(static_cast<bool>(statistics.admissions == 2));
    check(static_cast<bool>(statistics.stores == 2));
    check(static_cast<bool>(statistics.hits == 2));

    auto filtered_inner = std::make_shared<TestExpertVictimCache>(1024);
    auto filtered = create_reuse_victim_cache(filtered_inner, 4);
    check(static_cast<bool>(filtered));
    for (uint32_t index = 0; index < 8; ++index)
    {
        filtered->admit("weak." + std::to_string(index), gate_up, down);
    }
    for (uint32_t index = 0; index < 2; ++index)
    {
        filtered->admit("weak." + std::to_string(index + 1), gate_up, down);
    }
    const ExpertVictimCacheStatistics filtered_statistics = filtered->statistics();
    check(static_cast<bool>(filtered_statistics.admissions == 4));
    check(static_cast<bool>(filtered_statistics.filtered_admissions == 6));
    check(static_cast<bool>(filtered_statistics.reused_admissions == 2));
    check(static_cast<bool>(filtered_statistics.probe_admissions == 2));
    check(static_cast<bool>(filtered->restore("weak.0", *gate_up, *down)));
    check(static_cast<bool>(filtered->restore("weak.1", *gate_up, *down)));
    check(static_cast<bool>(filtered->restore("weak.2", *gate_up, *down)));
    check(static_cast<bool>(!filtered->restore("weak.3", *gate_up, *down)));
}

void test_mapped_file_range_and_shared_buffer()
{
    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_mapped_file_test_");
    const std::filesystem::path path = directory / "range.bin";
    const size_t file_size = 3 * 4096 + 257;
    std::vector<uint8_t> expected(file_size);
    for (size_t index = 0; index < expected.size(); ++index)
        expected[index] = static_cast<uint8_t>((index * 29 + 7) % 251);
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(expected.data()), static_cast<std::streamsize>(expected.size()));
    }

    {
        constexpr size_t offset = 123;
        constexpr size_t byte_count = 2 * 4096 + 91;
        auto mapped = MappedFileRange::open(path, offset, byte_count);
        check(static_cast<bool>(mapped));
        mapped.value()->prefault();
        check(static_cast<bool>(mapped.value()->size() == byte_count));
        check(static_cast<bool>(mapped.value()->data()[0] == expected[offset]));
        check(static_cast<bool>(mapped.value()->data()[byte_count - 1] == expected[offset + byte_count - 1]));

        MxFp4ByteBuffer shared = mapped.value()->share_bytes();
        check(static_cast<bool>(shared.size() == byte_count));
        check(static_cast<bool>(shared.front() == expected[offset]));
        check(static_cast<bool>(shared.back() == expected[offset + byte_count - 1]));
        MxFp4ByteBuffer copy = shared;
        copy.front() ^= 0xff;
        check(static_cast<bool>(shared.front() == expected[offset]));
        check(static_cast<bool>(copy.front() != shared.front()));

        auto truncated = MappedFileRange::open(path, file_size - 8, 16);
        check(static_cast<bool>(!truncated));
        check(static_cast<bool>(truncated.error().code == ErrorCode::InvalidModel));
    }

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_safetensors_dense_mmap()
{
    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_safetensors_mmap_test_");
    const std::filesystem::path path = directory / "model.safetensors";
    std::string header = R"({"bf16":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"f32":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})";
    while (header.size() % 8 != 0)
        header.push_back(' ');
    const uint64_t header_size = header.size();
    const std::array<uint16_t, 2> bfloat_values = {float_to_bfloat16(1.0f), float_to_bfloat16(-2.0f)};
    const std::array<float, 2> float_values = {3.5f, -2.25f};
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
        stream.write(reinterpret_cast<const char*>(bfloat_values.data()), sizeof(bfloat_values));
        stream.write(reinterpret_cast<const char*>(float_values.data()), sizeof(float_values));
    }

    {
        auto archive = SafetensorsArchive::open(directory);
        check(static_cast<bool>(archive));
        auto bfloat = archive.value().load_tensor("bf16");
        check(static_cast<bool>(bfloat));
        check(static_cast<bool>(bfloat.value().mapped_data));
        check(static_cast<bool>(bfloat.value().bfloat16_data.empty()));
        check(static_cast<bool>(bfloat.value().bfloat16_values().size() == 2));
        check_near(bfloat16_to_float(bfloat.value().bfloat16_values()[1]), -2.0f, 0.0f);

        auto floating = archive.value().load_tensor("f32");
        check(static_cast<bool>(floating));
        check(static_cast<bool>(floating.value().mapped_data));
        check(static_cast<bool>(floating.value().float32_data.empty()));
        check(static_cast<bool>(floating.value().float32_values().size() == 2));
        check_near(floating.value().float32_values()[0], 3.5f, 0.0f);

        auto slice = archive.value().load_bfloat16_slice("bf16", 1, {1});
        check(static_cast<bool>(slice));
        check(static_cast<bool>(slice.value().mapped_data));
        check_near(bfloat16_to_float(slice.value().bfloat16_values()[0]), -2.0f, 0.0f);
    }

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_safetensors_packed_mxfp4_expert()
{
    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_safetensors_packed_test_");
    const auto write_archive = [](const std::filesystem::path& path, std::string header, const std::vector<uint8_t>& data) {
        while (header.size() % 8 != 0)
            header.push_back(' ');
        const uint64_t header_size = header.size();
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
        stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    };
    write_archive(directory / "model.safetensors", R"({"gate_blocks":{"dtype":"U8","shape":[1,1,1,16],"data_offsets":[0,16]},"gate_scales":{"dtype":"U8","shape":[1,1,1],"data_offsets":[16,17]}})", std::vector<uint8_t>(17, 3));
    std::vector<uint8_t> packed(17, 9);
    packed.back() = 123;
    write_archive(directory / "ncnn-moe-packed-experts.safetensors",
                  R"({"__ncnn_moe_packed__.0.gate_blocks":{"dtype":"U8","shape":[1,1,16],"data_offsets":[0,16]},"__ncnn_moe_packed__.0.gate_scales":{"dtype":"U8","shape":[1,1],"data_offsets":[16,17]}})", packed);

    auto archive = SafetensorsArchive::open(directory);
    check(static_cast<bool>(archive));
    auto expert = archive.value().load_mxfp4_expert("gate_blocks", "gate_scales", 0, 1, 32, SafetensorLoadDeferMxfp4Data);
    check(static_cast<bool>(expert));
    check(static_cast<bool>(expert.value().mxfp4_file_storage));
    const MxFp4FileStorage& storage = *expert.value().mxfp4_file_storage;
    check(static_cast<bool>(storage.blocks_path == storage.scales_path));
    check(static_cast<bool>(storage.blocks_offset + storage.blocks_bytes == storage.scales_offset));

    Mxfp4ExpertCache cache(34, 1, {}, ExpertCacheBufferedReads);
    auto pair = cache.acquire_pair(expert.value(), expert.value());
    check(static_cast<bool>(pair));
    check(static_cast<bool>(pair.value().gate_up->mxfp4_blocks.front() == 9));
    check(static_cast<bool>(pair.value().gate_up->mxfp4_scales.front() == 123));

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_file_backed_mxfp4_expert_cache()
{
    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_expert_cache_test_");
    const std::filesystem::path blocks_path = directory / "blocks.bin";
    const std::filesystem::path scales_path = directory / "scales.bin";
    {
        std::vector<uint8_t> blocks(64);
        for (size_t index = 0; index < blocks.size(); ++index)
            blocks[index] = static_cast<uint8_t>(index);
        std::ofstream stream(blocks_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(blocks.data()), static_cast<std::streamsize>(blocks.size()));
    }
    {
        const std::vector<uint8_t> scales = {101, 102, 103, 104};
        std::ofstream stream(scales_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(scales.data()), static_cast<std::streamsize>(scales.size()));
    }

    auto file_backed = [&](uint64_t block_offset, uint64_t scale_offset, uint64_t block_bytes = 16) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {1, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = blocks_path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_bytes = block_bytes;
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
        const std::string prepared_key = Mxfp4ExpertCache::make_pair_key(gate_zero, down_zero);
        check(cache.is_ready(gate_zero, down_zero, prepared_key));
        auto hit = cache.acquire_pair(gate_zero, down_zero, std::numeric_limits<uint32_t>::max(), prepared_key);
        check(static_cast<bool>(hit));
        check(static_cast<bool>(hit.value().cache_hit));
        check(static_cast<bool>(hit.value().bytes_read == 0));
    }
    {
        const std::string prepared_key = Mxfp4ExpertCache::make_pair_key(gate_zero, down_zero);
        const std::array<ExpertCachePairRequest, 1> requests = {{
            &gate_zero,
            &down_zero,
            std::numeric_limits<uint32_t>::max(),
            prepared_key,
        }};
        std::array<ExpertCacheLease, 1> leases;
        auto acquired = cache.try_acquire_ready_pairs(requests, leases);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value()));
        check(static_cast<bool>(leases[0].cache_hit));
        check(static_cast<bool>(leases[0].bytes_read == 0));
        check(static_cast<bool>(leases[0].gate_up->mxfp4_blocks.front() == 0));
    }
    {
        auto second = cache.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(second));
        check(static_cast<bool>(!second.value().cache_hit));
        check(static_cast<bool>(second.value().gate_up->mxfp4_blocks.front() == 32));
        check(static_cast<bool>(second.value().down->mxfp4_scales.front() == 104));
    }
    const ExpertCacheStatistics statistics = cache.statistics();
    check(static_cast<bool>(statistics.hits == 2));
    check(static_cast<bool>(statistics.misses == 2));
    check(static_cast<bool>(statistics.evictions == 1));
    check(static_cast<bool>(statistics.bytes_read == 68));
    check(static_cast<bool>(statistics.resident_bytes == 34));
    check(static_cast<bool>(statistics.mapped_ranges == 8));
    check(static_cast<bool>(statistics.mapped_bytes == statistics.bytes_read));

    const std::filesystem::path packed_path = directory / "packed.bin";
    {
        std::vector<uint8_t> packed;
        packed.reserve(34);
        for (uint8_t value = 0; value < 16; ++value)
            packed.push_back(value);
        packed.push_back(101);
        for (uint8_t value = 16; value < 32; ++value)
            packed.push_back(value);
        packed.push_back(102);
        std::ofstream stream(packed_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    }
    auto packed_tensor = [&](uint64_t block_offset, uint64_t scale_offset) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {1, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = packed_path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_bytes = 16;
        storage->scales_path = packed_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_bytes = 1;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    const TensorData packed_gate = packed_tensor(0, 16);
    const TensorData packed_down = packed_tensor(17, 33);
    Mxfp4ExpertCache packed_cache(34, 1, {}, ExpertCacheBufferedReads);
    auto packed_pair = packed_cache.acquire_pair(packed_gate, packed_down);
    check(static_cast<bool>(packed_pair));
    check(static_cast<bool>(packed_pair.value().gate_up->mxfp4_blocks.front() == 0));
    check(static_cast<bool>(packed_pair.value().gate_up->mxfp4_scales.front() == 101));
    check(static_cast<bool>(packed_pair.value().down->mxfp4_blocks.front() == 16));
    check(static_cast<bool>(packed_pair.value().down->mxfp4_scales.front() == 102));
    const ExpertCacheStatistics packed_statistics = packed_cache.statistics();
    check(static_cast<bool>(packed_statistics.buffered_read_ranges == 1));
    check(static_cast<bool>(packed_statistics.buffered_read_bytes == 34));

    const std::filesystem::path coalesced_path = directory / "coalesced.bin";
    {
        std::vector<uint8_t> bytes(68);
        for (size_t index = 0; index < bytes.size(); ++index)
            bytes[index] = static_cast<uint8_t>(index);
        std::ofstream stream(coalesced_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    auto coalesced_tensor = [&](uint64_t block_offset, uint64_t scale_offset) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {1, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = coalesced_path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_bytes = 16;
        storage->scales_path = coalesced_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_bytes = 1;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    const TensorData coalesced_gate_zero = coalesced_tensor(0, 16);
    const TensorData coalesced_down_zero = coalesced_tensor(17, 33);
    const TensorData coalesced_gate_one = coalesced_tensor(34, 50);
    const TensorData coalesced_down_one = coalesced_tensor(51, 67);
    Mxfp4ExpertCache coalesced_cache(
        68,
        1,
        {},
        ExpertCacheBufferedReads | ExpertCacheCrossExpertReadCoalescing);
    auto coalesced_request_zero = coalesced_cache.request_pair(coalesced_gate_zero, coalesced_down_zero, 0);
    auto coalesced_request_one = coalesced_cache.request_pair(coalesced_gate_one, coalesced_down_one, 1);
    check(static_cast<bool>(coalesced_request_zero && !coalesced_request_zero.value()));
    check(static_cast<bool>(coalesced_request_one && !coalesced_request_one.value()));
    coalesced_cache.wait_for_background_work();
    auto coalesced_zero = coalesced_cache.acquire_pair(coalesced_gate_zero, coalesced_down_zero, 0);
    auto coalesced_one = coalesced_cache.acquire_pair(coalesced_gate_one, coalesced_down_one, 1);
    check(static_cast<bool>(coalesced_zero));
    check(static_cast<bool>(coalesced_one));
    check(coalesced_zero.value().gate_up->mxfp4_blocks.front() == 0);
    check(coalesced_zero.value().down->mxfp4_blocks.front() == 17);
    check(coalesced_one.value().gate_up->mxfp4_blocks.front() == 34);
    check(coalesced_one.value().down->mxfp4_scales.front() == 67);
    const ExpertCacheStatistics coalesced_statistics = coalesced_cache.statistics();
    check(coalesced_statistics.buffered_read_ranges == 1);
    check(coalesced_statistics.buffered_read_bytes == 68);
    check(coalesced_statistics.coalesced_read_batches == 1);
    check(coalesced_statistics.coalesced_experts == 2);
    check(coalesced_statistics.coalesced_read_ranges_saved == 1);

    Mxfp4ExpertCache resolved_predictions(68, 1, {}, ExpertCacheBufferedReads);
    auto prediction_zero = resolved_predictions.prefetch_pair(gate_zero, down_zero, 1, "prediction-zero");
    auto prediction_one = resolved_predictions.prefetch_pair(gate_one, down_one, 1, "prediction-one");
    check(static_cast<bool>(prediction_zero && !prediction_zero.value()));
    check(static_cast<bool>(prediction_one && !prediction_one.value()));
    resolved_predictions.wait_for_background_work();
    const std::array<std::string_view, 1> demanded_prediction = {"prediction-zero"};
    resolved_predictions.resolve_predictions(1, demanded_prediction);
    check(resolved_predictions.is_ready(gate_zero, down_zero, "prediction-zero"));
    check(!resolved_predictions.is_ready(gate_one, down_one, "prediction-one"));
    check(resolved_predictions.statistics().unused_speculative_reads == 1);

    Mxfp4ExpertCache forward_aware(
        68,
        1,
        {},
        ExpertCacheBufferedReads | ExpertCacheForwardAwareEviction,
        4);
    {
        auto group_zero = forward_aware.acquire_pair(gate_zero, down_zero, 0, "forward-zero");
        auto group_two = forward_aware.acquire_pair(gate_one, down_one, 2, "forward-two");
        check(static_cast<bool>(group_zero));
        check(static_cast<bool>(group_two));
    }
    auto group_one = forward_aware.acquire_pair(gate_zero, down_one, 1, "forward-one");
    check(static_cast<bool>(group_one));
    check(!forward_aware.is_ready(gate_zero, down_zero, "forward-zero"));
    check(forward_aware.is_ready(gate_one, down_one, "forward-two"));

    Mxfp4ExpertCache predicted_protection(
        68,
        1,
        {},
        ExpertCacheBufferedReads | ExpertCacheForwardAwareEviction,
        4);
    {
        auto exact = predicted_protection.acquire_pair(
            gate_zero,
            down_zero,
            2,
            "protected-exact");
        auto repeated = predicted_protection.acquire_pair(
            gate_zero,
            down_zero,
            2,
            "protected-exact");
        check(static_cast<bool>(exact));
        check(static_cast<bool>(repeated));
    }
    auto predicted = predicted_protection.prefetch_pair(
        gate_one,
        down_one,
        1,
        "protected-prediction");
    check(static_cast<bool>(predicted && !predicted.value()));
    predicted_protection.wait_for_background_work();
    auto incoming = predicted_protection.acquire_pair(
        gate_zero,
        down_one,
        0,
        "protected-incoming");
    check(static_cast<bool>(incoming));
    check(!predicted_protection.is_ready(
        gate_zero,
        down_zero,
        "protected-exact"));
    check(predicted_protection.is_ready(
        gate_one,
        down_one,
        "protected-prediction"));

    const std::filesystem::path clustered_path = directory / "clustered.bin";
    {
        std::vector<uint8_t> clustered(112, UINT8_C(0xee));
        clustered[0] = 101;
        clustered[1] = 102;
        clustered[2] = 201;
        clustered[3] = 202;
        clustered[4] = 111;
        clustered[5] = 112;
        for (uint8_t value = 0; value < 96; ++value)
            clustered[16 + value] = value;
        std::ofstream stream(clustered_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(clustered.data()), static_cast<std::streamsize>(clustered.size()));
    }
    TensorData clustered_gate;
    clustered_gate.dtype = DType::MxFp4;
    clustered_gate.shape = {4, 32};
    auto clustered_gate_storage = std::make_shared<MxFp4FileStorage>();
    clustered_gate_storage->blocks_path = clustered_path.string();
    clustered_gate_storage->blocks_offset = 16;
    clustered_gate_storage->blocks_bytes = 32;
    clustered_gate_storage->scales_path = clustered_path.string();
    clustered_gate_storage->scales_offset = 0;
    clustered_gate_storage->scales_bytes = 2;
    clustered_gate_storage->secondary_blocks_path = clustered_path.string();
    clustered_gate_storage->secondary_blocks_offset = 80;
    clustered_gate_storage->secondary_blocks_bytes = 32;
    clustered_gate_storage->secondary_scales_path = clustered_path.string();
    clustered_gate_storage->secondary_scales_offset = 4;
    clustered_gate_storage->secondary_scales_bytes = 2;
    clustered_gate_storage->interleave_rows = true;
    clustered_gate.mxfp4_file_storage = std::move(clustered_gate_storage);
    TensorData clustered_down;
    clustered_down.dtype = DType::MxFp4;
    clustered_down.shape = {2, 32};
    auto clustered_down_storage = std::make_shared<MxFp4FileStorage>();
    clustered_down_storage->blocks_path = clustered_path.string();
    clustered_down_storage->blocks_offset = 48;
    clustered_down_storage->blocks_bytes = 32;
    clustered_down_storage->scales_path = clustered_path.string();
    clustered_down_storage->scales_offset = 2;
    clustered_down_storage->scales_bytes = 2;
    clustered_down.mxfp4_file_storage = std::move(clustered_down_storage);
    Mxfp4ExpertCache clustered_cache(102, 1, {}, ExpertCacheBufferedReads);
    auto clustered_pair = clustered_cache.acquire_pair(clustered_gate, clustered_down);
    check(static_cast<bool>(clustered_pair));
    check(clustered_pair.value().gate_up->mxfp4_blocks[0] == 0);
    check(clustered_pair.value().gate_up->mxfp4_blocks[16] == 64);
    check(clustered_pair.value().gate_up->mxfp4_blocks[32] == 16);
    check(clustered_pair.value().gate_up->mxfp4_blocks[48] == 80);
    check(clustered_pair.value().gate_up->mxfp4_scales[0] == 101);
    check(clustered_pair.value().gate_up->mxfp4_scales[1] == 111);
    check(clustered_pair.value().gate_up->mxfp4_scales[2] == 102);
    check(clustered_pair.value().gate_up->mxfp4_scales[3] == 112);
    check(clustered_pair.value().down->mxfp4_blocks[0] == 32);
    check(clustered_pair.value().down->mxfp4_blocks[16] == 48);
    check(clustered_pair.value().down->mxfp4_scales[0] == 201);
    check(clustered_pair.value().down->mxfp4_scales[1] == 202);
    const ExpertCacheStatistics clustered_statistics = clustered_cache.statistics();
    check(clustered_statistics.buffered_read_ranges == 2);
    check(clustered_statistics.buffered_read_bytes == 102);

    const std::filesystem::path fragmented_path = directory / "fragmented.bin";
    {
        std::vector<uint8_t> fragmented(107, UINT8_C(0xee));
        for (uint8_t value = 0; value < 32; ++value)
            fragmented[value] = value;
        fragmented[33] = 101;
        fragmented[34] = 102;
        for (uint8_t value = 0; value < 32; ++value)
            fragmented[36 + value] = static_cast<uint8_t>(64 + value);
        fragmented[69] = 111;
        fragmented[70] = 112;
        for (uint8_t value = 0; value < 32; ++value)
            fragmented[72 + value] = static_cast<uint8_t>(32 + value);
        fragmented[105] = 201;
        fragmented[106] = 202;
        std::ofstream stream(fragmented_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(fragmented.data()), static_cast<std::streamsize>(fragmented.size()));
    }
    TensorData fragmented_gate;
    fragmented_gate.dtype = DType::MxFp4;
    fragmented_gate.shape = {4, 32};
    auto fragmented_gate_storage = std::make_shared<MxFp4FileStorage>();
    fragmented_gate_storage->blocks_path = fragmented_path.string();
    fragmented_gate_storage->blocks_offset = 0;
    fragmented_gate_storage->blocks_bytes = 32;
    fragmented_gate_storage->scales_path = fragmented_path.string();
    fragmented_gate_storage->scales_offset = 33;
    fragmented_gate_storage->scales_bytes = 2;
    fragmented_gate_storage->secondary_blocks_path = fragmented_path.string();
    fragmented_gate_storage->secondary_blocks_offset = 36;
    fragmented_gate_storage->secondary_blocks_bytes = 32;
    fragmented_gate_storage->secondary_scales_path = fragmented_path.string();
    fragmented_gate_storage->secondary_scales_offset = 69;
    fragmented_gate_storage->secondary_scales_bytes = 2;
    fragmented_gate_storage->interleave_rows = true;
    fragmented_gate.mxfp4_file_storage = std::move(fragmented_gate_storage);
    TensorData fragmented_down;
    fragmented_down.dtype = DType::MxFp4;
    fragmented_down.shape = {2, 32};
    auto fragmented_down_storage = std::make_shared<MxFp4FileStorage>();
    fragmented_down_storage->blocks_path = fragmented_path.string();
    fragmented_down_storage->blocks_offset = 72;
    fragmented_down_storage->blocks_bytes = 32;
    fragmented_down_storage->scales_path = fragmented_path.string();
    fragmented_down_storage->scales_offset = 105;
    fragmented_down_storage->scales_bytes = 2;
    fragmented_down.mxfp4_file_storage = std::move(fragmented_down_storage);
    Mxfp4ExpertCache fragmented_cache(102, 1, {}, ExpertCacheBufferedReads);
    auto fragmented_pair = fragmented_cache.acquire_pair(fragmented_gate, fragmented_down);
    check(static_cast<bool>(fragmented_pair));
    check(fragmented_pair.value().gate_up->mxfp4_blocks[0] == 0);
    check(fragmented_pair.value().gate_up->mxfp4_blocks[16] == 64);
    check(fragmented_pair.value().gate_up->mxfp4_blocks[32] == 16);
    check(fragmented_pair.value().gate_up->mxfp4_blocks[48] == 80);
    check(fragmented_pair.value().gate_up->mxfp4_scales[0] == 101);
    check(fragmented_pair.value().gate_up->mxfp4_scales[1] == 111);
    check(fragmented_pair.value().down->mxfp4_blocks[0] == 32);
    check(fragmented_pair.value().down->mxfp4_scales[0] == 201);
    const ExpertCacheStatistics fragmented_statistics = fragmented_cache.statistics();
    check(fragmented_statistics.buffered_read_ranges == 6);
    check(fragmented_statistics.buffered_read_bytes == 102);

    TensorData invalid_secondary = packed_gate;
    auto invalid_secondary_storage = std::make_shared<MxFp4FileStorage>(*packed_gate.mxfp4_file_storage);
    invalid_secondary_storage->secondary_blocks_path = packed_path.string();
    invalid_secondary_storage->secondary_blocks_offset = 17;
    invalid_secondary_storage->secondary_blocks_bytes = 16;
    invalid_secondary.mxfp4_file_storage = std::move(invalid_secondary_storage);
    Mxfp4ExpertCache invalid_secondary_cache(50, 1, {}, ExpertCacheBufferedReads);
    auto invalid_secondary_pair = invalid_secondary_cache.acquire_pair(invalid_secondary, packed_down);
    check(!invalid_secondary_pair);
    check(invalid_secondary_pair.error().code == ErrorCode::InvalidModel);

    auto victim = std::make_shared<TestExpertVictimCache>(68);
    Mxfp4ExpertCache tiered(34, 1, victim, ExpertCacheMemoryMapRanges);
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

    if (NcnnLinearOperator::vulkan_device_count() > 0)
    {
        auto gpu_source = cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(gpu_source));
        auto gpu_victim = create_vulkan_victim_cache(68);
        check(static_cast<bool>(gpu_victim));
        gpu_victim->admit("gpu-roundtrip", gpu_source.value().gate_up, gpu_source.value().down);
        gpu_victim->wait_for_background_work();
        check(static_cast<bool>(gpu_victim->statistics().stores == 1));
        auto gpu_restored = gpu_victim->restore("gpu-roundtrip", gate_zero, down_zero);
        check(static_cast<bool>(gpu_restored));
        check(static_cast<bool>(gpu_restored->gate_up->mxfp4_blocks.front() == 0));
        check(static_cast<bool>(gpu_restored->down->mxfp4_blocks.front() == 16));
        check(static_cast<bool>(gpu_restored->gate_up->mxfp4_scales.front() == 101));
        check(static_cast<bool>(gpu_restored->down->mxfp4_scales.front() == 102));
        const ExpertVictimCacheStatistics gpu_statistics = gpu_victim->statistics();
        check(static_cast<bool>(gpu_statistics.bytes_uploaded >= 34));
        check(static_cast<bool>(gpu_statistics.bytes_downloaded == gpu_statistics.bytes_uploaded));
        check(static_cast<bool>(gpu_statistics.mapped_stores <= gpu_statistics.stores));
        check(static_cast<bool>(gpu_statistics.mapped_restores <= gpu_statistics.hits));
        check(static_cast<bool>(gpu_statistics.mapped_stores == gpu_statistics.mapped_restores));
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

    {
        auto delayed = std::make_shared<DelayedExpertVictimCache>();
        Mxfp4ExpertCache any_ready(68, 2, delayed);
        const std::array<ExpertCachePairRequest, 2> requests = {{
            {
                &gate_zero,
                &down_zero,
                0,
                "slow",
            },
            {
                &gate_one,
                &down_one,
                0,
                "fast",
            },
        }};
        std::array<ExpertCacheLease, 2> leases;
        auto acquired = any_ready.wait_acquire_ready_pairs(requests, leases);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value() == 1));
        check(static_cast<bool>(!leases[0].gate_up));
        check(static_cast<bool>(leases[1].gate_up));
        check(static_cast<bool>(leases[1].gate_up->mxfp4_blocks.front() == 7));

        const std::array<ExpertCachePairRequest, 1> slow_request = {{
            {
                &gate_zero,
                &down_zero,
                0,
                "slow",
            },
        }};
        std::array<ExpertCacheLease, 1> slow_lease;
        acquired = any_ready.wait_acquire_ready_pairs(slow_request, slow_lease);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value() == 1));
        check(static_cast<bool>(slow_lease[0].gate_up));
        check(static_cast<bool>(any_ready.statistics().misses == 2));

        Mxfp4ExpertCache front_ready(68, 2, delayed);
        leases = {};
        acquired = front_ready.wait_acquire_ready_pairs(requests, leases, false);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value() == 2));
        check(static_cast<bool>(leases[0].gate_up));
        check(static_cast<bool>(leases[1].gate_up));
    }

    Mxfp4ExpertCache speculative(68, 1);
    check(static_cast<bool>(speculative.prefetch_pair(gate_zero, down_zero)));
    speculative.wait_for_background_work();
    check(speculative.is_ready(gate_zero, down_zero));
    auto prefetched = speculative.acquire_pair(gate_zero, down_zero);
    check(static_cast<bool>(prefetched));
    check(static_cast<bool>(speculative.statistics().speculative_reads == 1));
    check(static_cast<bool>(speculative.statistics().queued_reads == 1));

    Mxfp4ExpertCache adaptive_replacement(68, 1);
    {
        auto first = adaptive_replacement.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(first));
    }
    {
        auto reused = adaptive_replacement.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(reused && reused.value().cache_hit));
    }
    {
        auto single_use = adaptive_replacement.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(single_use));
    }
    {
        auto third = adaptive_replacement.acquire_pair(gate_zero, down_one);
        check(static_cast<bool>(third));
    }
    {
        auto retained_hot = adaptive_replacement.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(retained_hot && retained_hot.value().cache_hit));
    }
    {
        auto recent_ghost = adaptive_replacement.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(recent_ghost));
        const ExpertCacheStatistics adapted = adaptive_replacement.statistics();
        check(static_cast<bool>(adapted.arc_recent_ghost_hits == 1));
        check(static_cast<bool>(adapted.arc_recent_target_bytes == 34));
    }
    {
        auto frequent_ghost = adaptive_replacement.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(frequent_ghost));
        const ExpertCacheStatistics adapted = adaptive_replacement.statistics();
        check(static_cast<bool>(adapted.arc_frequent_ghost_hits == 1));
        check(static_cast<bool>(adapted.arc_recent_target_bytes == 0));
        check(static_cast<bool>(adapted.arc_recent_bytes + adapted.arc_frequent_bytes == adapted.resident_bytes));
    }

    Mxfp4ExpertCache pressure(34, 1);
    auto pinned = pressure.acquire_pair(gate_zero, down_zero);
    check(static_cast<bool>(pinned));
    check(static_cast<bool>(pressure.prefetch_pair(gate_one, down_one)));
    check(static_cast<bool>(pressure.statistics().speculative_reads == 0));
    check(static_cast<bool>(pressure.statistics().dropped_speculative_admissions == 1));
    auto exhausted = pressure.acquire_pair(gate_one, down_one);
    check(static_cast<bool>(!exhausted));
    check(static_cast<bool>(exhausted.error().code == ErrorCode::InvalidArgument));

    Mxfp4ExpertCache variable_size(52, 1);
    const TensorData small_gate = file_backed(0, 0, 8);
    const TensorData small_down = file_backed(8, 1, 8);
    {
        auto small = variable_size.acquire_pair(small_gate, small_down);
        check(static_cast<bool>(small));
        check(static_cast<bool>(small.value().bytes_read == 18));
    }
    {
        auto regular = variable_size.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(regular));
        check(static_cast<bool>(regular.value().bytes_read == 34));
    }
    const ExpertCacheStatistics variable_statistics = variable_size.statistics();
    check(static_cast<bool>(variable_statistics.resident_bytes == 52));
    check(static_cast<bool>(variable_statistics.arc_recent_bytes + variable_statistics.arc_frequent_bytes == variable_statistics.resident_bytes));

    Mxfp4ExpertCache oversized_prefetch(17, 1);
    check(static_cast<bool>(oversized_prefetch.prefetch_pair(small_gate, small_down)));
    check(static_cast<bool>(oversized_prefetch.statistics().dropped_speculative_admissions == 1));

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
        stream.write(reinterpret_cast<const char*>(blocks.data()), static_cast<std::streamsize>(blocks.size()));
    }
    {
        std::vector<uint8_t> scales(large_scales_per_tensor * 2);
        for (size_t index = 0; index < scales.size(); ++index)
            scales[index] = static_cast<uint8_t>((index * 7) % 239);
        std::ofstream stream(large_scales_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(scales.data()), static_cast<std::streamsize>(scales.size()));
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
    const TensorData large_down = large_file_backed(large_blocks_per_tensor, large_scales_per_tensor);
    const uint64_t large_pair_bytes = (large_blocks_per_tensor + large_scales_per_tensor) * 2;
    Mxfp4ExpertCache large_reads(large_pair_bytes, 2, {}, true);
    auto large_pair = large_reads.acquire_pair(large_gate, large_down);
    check(static_cast<bool>(large_pair));
    check(static_cast<bool>(large_pair.value().bytes_read == large_pair_bytes));
    check(static_cast<bool>(large_reads.statistics().mapped_ranges == 4));
    check(static_cast<bool>(large_reads.statistics().mapped_bytes == large_pair_bytes));
    check(static_cast<bool>(large_pair.value().gate_up->mxfp4_blocks.front() == 0));
    check(static_cast<bool>(large_pair.value().gate_up->mxfp4_blocks.back() == static_cast<uint8_t>(((large_blocks_per_tensor - 1) * 13) % 251)));
    check(static_cast<bool>(large_pair.value().down->mxfp4_scales.back() == static_cast<uint8_t>(((large_scales_per_tensor * 2 - 1) * 7) % 239)));

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_cpu_topology_parsing_and_partitioning()
{
    check(static_cast<bool>(parse_linux_cpu_list("0-3,8,10-11") == std::vector<uint32_t>({0, 1, 2, 3, 8, 10, 11})));
    check(static_cast<bool>(parse_linux_cpu_list("4,2-4,2") == std::vector<uint32_t>({2, 3, 4})));
    check(static_cast<bool>(parse_linux_cpu_list("3-1").empty()));
    check(static_cast<bool>(parse_linux_cpu_list("1,,2").empty()));
    check(static_cast<bool>(parse_linux_cpu_list("1,").empty()));

    CpuTopology flat;
    flat.allowed_cpus = {2, 4, 7, 9, 12};
    const std::vector<std::vector<uint32_t>> flat_partitions = partition_cpu_topology(flat, 2);
    check(static_cast<bool>(flat_partitions.size() == 2));
    check(static_cast<bool>(flat_partitions[0] == std::vector<uint32_t>({2, 4, 7})));
    check(static_cast<bool>(flat_partitions[1] == std::vector<uint32_t>({9, 12})));

    CpuTopology numa;
    numa.allowed_cpus = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    numa.numa_nodes = {
        {0, 1, 2, 3},
        {4, 5, 6, 7, 8, 9, 10, 11},
    };
    const std::vector<std::vector<uint32_t>> numa_partitions = partition_cpu_topology(numa, 4);
    check(static_cast<bool>(numa_partitions.size() == 4));
    std::vector<uint32_t> partitioned_cpus;
    for (const std::vector<uint32_t>& partition : numa_partitions)
    {
        check(static_cast<bool>(!partition.empty()));
        const bool first_node = partition.front() < 4;
        for (uint32_t cpu : partition)
            check(static_cast<bool>((cpu < 4) == first_node));
        partitioned_cpus.insert(partitioned_cpus.end(), partition.begin(), partition.end());
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
    options.flags = SchedulerOptionForceStagedBatching;
    auto scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(scheduler));
    auto prefill_first = runtime.create_session(model.value());
    auto prefill_second = runtime.create_session(model.value());
    auto prefill_reference_first = runtime.create_session(model.value());
    auto prefill_reference_second = runtime.create_session(model.value());
    check(static_cast<bool>(prefill_first));
    check(static_cast<bool>(prefill_second));
    check(static_cast<bool>(prefill_reference_first));
    check(static_cast<bool>(prefill_reference_second));
    const std::vector<int32_t> first_prompt = {0, 1};
    const std::vector<int32_t> second_prompt = {1};
    auto prefill_future = scheduler.value()->submit_prefill({
        {prefill_first.value(), first_prompt},
        {prefill_second.value(), second_prompt},
    });
    auto prefill_results = prefill_future.get();
    auto first_reference = prefill_reference_first.value()->prefill(first_prompt);
    auto second_reference = prefill_reference_second.value()->prefill(second_prompt);
    check(static_cast<bool>(prefill_results.size() == 2));
    check(static_cast<bool>(prefill_results[0]));
    check(static_cast<bool>(prefill_results[1]));
    check(static_cast<bool>(first_reference));
    check(static_cast<bool>(second_reference));
    check(static_cast<bool>(
        prefill_first.value()->sequence_length() == 2));
    check(static_cast<bool>(
        prefill_second.value()->sequence_length() == 1));
    for (size_t index = 0;
         index < first_reference.value().logits.values.size();
         ++index)
    {
        check_near(
            prefill_results[0].value().logits.values[index],
            first_reference.value().logits.values[index],
            1e-5f);
    }
    for (size_t index = 0;
         index < second_reference.value().logits.values.size();
         ++index)
    {
        check_near(
            prefill_results[1].value().logits.values[index],
            second_reference.value().logits.values[index],
            1e-5f);
    }
    auto future = scheduler.value()->submit_decode({
        {first.value(), 0},
        {second.value(), 0},
    });
    std::vector<Result<DecodeResult>> results = future.get();
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
    std::vector<Result<DecodeResult>> ordered_first_result = ordered_first.get();
    std::vector<Result<DecodeResult>> ordered_second_result = ordered_second.get();
    check(static_cast<bool>(ordered_first_result[0]));
    check(static_cast<bool>(ordered_second_result[0]));
    check(static_cast<bool>(ordered_first_result[0].value().sequence_length == 1));
    check(static_cast<bool>(ordered_second_result[0].value().sequence_length == 2));
    for (size_t index = 0; index < reference_second.value().logits.values.size(); ++index)
    {
        check_near(ordered_second_result[0].value().logits.values[index], reference_second.value().logits.values[index], 1e-5f);
    }

    auto duplicate_future = scheduler.value()->submit_decode({
        {first.value(), 0},
        {first.value(), 1},
    });
    std::vector<Result<DecodeResult>> duplicate_results = duplicate_future.get();
    check(static_cast<bool>(!duplicate_results[0]));
    check(static_cast<bool>(!duplicate_results[1]));
    check(static_cast<bool>(first.value()->sequence_length() == 1));
    const SchedulerStatistics statistics = scheduler.value()->statistics();
    check(static_cast<bool>(statistics.worker_count == 2));
    check(static_cast<bool>(statistics.expert_threads_per_worker >= 1));
    check(static_cast<bool>(
        statistics.submitted_prefill_batches == 1));
    check(static_cast<bool>(
        statistics.submitted_prefill_requests == 2));
    check(static_cast<bool>(
        statistics.completed_prefill_requests == 2));
    check(static_cast<bool>(
        statistics.staged_prefill_batches == 1));
    check(static_cast<bool>(
        statistics.staged_prefill_requests == 2));
    check(static_cast<bool>(statistics.submitted_batches == 4));
    check(static_cast<bool>(statistics.submitted_requests == 6));
    check(static_cast<bool>(statistics.completed_requests == 6));
    check(static_cast<bool>(statistics.rejected_requests == 2));
    check(static_cast<bool>(statistics.max_batch_size == 2));
    check(static_cast<bool>(statistics.max_in_flight >= 2));
    check(static_cast<bool>(statistics.staged_batches == 1));
    check(static_cast<bool>(statistics.staged_requests == 2));
    check(static_cast<bool>(statistics.logical_expert_batches > statistics.physical_expert_batches));
    check(static_cast<bool>(statistics.coalesced_expert_batches > 0));
    check(static_cast<bool>(statistics.coalesced_expert_routes >= 2));
    check(static_cast<bool>(statistics.max_coalesced_expert_batch_size >= 2));

    auto bypass_first = runtime.create_session(model.value());
    auto bypass_second = runtime.create_session(model.value());
    check(static_cast<bool>(bypass_first));
    check(static_cast<bool>(bypass_second));
    SchedulerOptions adaptive_options;
    adaptive_options.worker_count = 2;
    adaptive_options.adaptive_probe_interval = 2;
    auto bypass_scheduler = runtime.create_scheduler(adaptive_options);
    check(static_cast<bool>(bypass_scheduler));
    auto bypass_future = bypass_scheduler.value()->submit_decode({
        {bypass_first.value(), 0},
        {bypass_second.value(), 1},
    });
    std::vector<Result<DecodeResult>> bypass_results = bypass_future.get();
    check(static_cast<bool>(bypass_results.size() == 2));
    check(static_cast<bool>(bypass_results[0]));
    check(static_cast<bool>(bypass_results[1]));
    const SchedulerStatistics bypass_statistics = bypass_scheduler.value()->statistics();
    check(static_cast<bool>(bypass_statistics.staged_batches == 0));
    check(static_cast<bool>(bypass_statistics.staging_bypassed_batches == 1));
    for (uint32_t iteration = 0; iteration < 3; ++iteration)
    {
        auto adaptive_future = bypass_scheduler.value()->submit_decode({
            {
                bypass_first.value(),
                0,
            },
            {
                bypass_second.value(),
                1,
            },
        });
        auto adaptive_results = adaptive_future.get();
        check(static_cast<bool>(adaptive_results.size() == 2));
        check(static_cast<bool>(adaptive_results[0]));
        check(static_cast<bool>(adaptive_results[1]));
    }
    const SchedulerStatistics adaptive_statistics = bypass_scheduler.value()->statistics();
    check(static_cast<bool>(adaptive_statistics.adaptive_independent_decisions + adaptive_statistics.adaptive_staged_decisions == 4));
    check(static_cast<bool>(adaptive_statistics.adaptive_independent_decisions >= 1));
    check(static_cast<bool>(adaptive_statistics.adaptive_staged_decisions >= 1));
    check(static_cast<bool>(adaptive_statistics.adaptive_probe_decisions >= 1));
    check(static_cast<bool>(adaptive_statistics.adaptive_independent_observations >= 1));
    check(static_cast<bool>(adaptive_statistics.adaptive_staged_observations >= 1));
    check(static_cast<bool>(adaptive_statistics.adaptive_resident_decisions + adaptive_statistics.adaptive_mixed_decisions + adaptive_statistics.adaptive_storage_decisions == adaptive_statistics.adaptive_independent_decisions + adaptive_statistics.adaptive_staged_decisions));
    check(static_cast<bool>(adaptive_statistics.adaptive_resident_observations + adaptive_statistics.adaptive_mixed_observations + adaptive_statistics.adaptive_storage_observations == adaptive_statistics.adaptive_independent_observations + adaptive_statistics.adaptive_staged_observations));
    check(static_cast<bool>(adaptive_statistics.adaptive_resident_observations >= 1));

    auto collected_first = runtime.create_session(model.value());
    auto collected_second = runtime.create_session(model.value());
    check(static_cast<bool>(collected_first));
    check(static_cast<bool>(collected_second));
    SchedulerOptions collection_options;
    collection_options.worker_count = 2;
    collection_options.cross_call_window_microseconds = 50000;
    collection_options.flags = SchedulerOptionForceStagedBatching;
    auto collection_scheduler = runtime.create_scheduler(collection_options);
    check(static_cast<bool>(collection_scheduler));
    auto collected_first_future = collection_scheduler.value()->submit_decode({
        {collected_first.value(), 0},
    });
    auto collected_second_future = collection_scheduler.value()->submit_decode({
        {collected_second.value(), 0},
    });
    const std::vector<Result<DecodeResult>> collected_first_result = collected_first_future.get();
    const std::vector<Result<DecodeResult>> collected_second_result = collected_second_future.get();
    check(static_cast<bool>(collected_first_result.size() == 1));
    check(static_cast<bool>(collected_second_result.size() == 1));
    check(static_cast<bool>(collected_first_result.front()));
    check(static_cast<bool>(collected_second_result.front()));
    const SchedulerStatistics collection_statistics = collection_scheduler.value()->statistics();
    check(static_cast<bool>(collection_statistics.submitted_batches == 2));
    check(static_cast<bool>(collection_statistics.submitted_requests == 2));
    check(static_cast<bool>(collection_statistics.cross_call_collected_batches == 1));
    check(static_cast<bool>(collection_statistics.cross_call_collected_requests == 2));
    check(static_cast<bool>(collection_statistics.max_cross_call_batch_size == 2));
    check(static_cast<bool>(collection_statistics.staged_batches == 1));
    check(static_cast<bool>(collection_statistics.staged_requests == 2));

    auto collection_backoff_session = runtime.create_session(model.value());
    check(static_cast<bool>(collection_backoff_session));
    SchedulerOptions collection_backoff_options;
    collection_backoff_options.worker_count = 2;
    collection_backoff_options.cross_call_window_microseconds = 1000;
    auto collection_backoff_scheduler = runtime.create_scheduler(collection_backoff_options);
    check(static_cast<bool>(collection_backoff_scheduler));
    for (int32_t token = 0; token < 6; ++token)
    {
        auto singleton_future = collection_backoff_scheduler.value()->submit_decode({
            {
                collection_backoff_session.value(),
                0,
            },
        });
        auto singleton = singleton_future.get();
        check(static_cast<bool>(singleton.size() == 1));
        check(static_cast<bool>(singleton.front()));
    }
    const SchedulerStatistics collection_backoff_statistics = collection_backoff_scheduler.value()->statistics();
    check(static_cast<bool>(collection_backoff_statistics.cross_call_collected_batches == 0));
    check(static_cast<bool>(collection_backoff_statistics.cross_call_collection_probes == 4));
    check(static_cast<bool>(collection_backoff_statistics.cross_call_collection_timeouts == 4));
    check(static_cast<bool>(collection_backoff_statistics.cross_call_collection_bypasses == 2));

    SchedulerOptions conflicting_staging;
    conflicting_staging.flags = SchedulerOptionDisableStagedBatching | SchedulerOptionForceStagedBatching;
    auto conflicting_scheduler = runtime.create_scheduler(conflicting_staging);
    check(static_cast<bool>(!conflicting_scheduler));
    check(static_cast<bool>(conflicting_scheduler.error().code == ErrorCode::InvalidArgument));

    SchedulerOptions excessive_probe_interval;
    excessive_probe_interval.adaptive_probe_interval = 1000001;
    auto invalid_adaptive_scheduler = runtime.create_scheduler(excessive_probe_interval);
    check(static_cast<bool>(!invalid_adaptive_scheduler));

    SchedulerOptions excessive_collection_window;
    excessive_collection_window.cross_call_window_microseconds = 1000001;
    check(static_cast<bool>(!runtime.create_scheduler(excessive_collection_window)));

    SchedulerOptions excessive_collection_batch;
    excessive_collection_batch.cross_call_max_batch_size = 1025;
    check(static_cast<bool>(!runtime.create_scheduler(excessive_collection_batch)));

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

    if (has_flag(runtime.capabilities().flags, RuntimeCapabilityVulkanCpuMix))
    {
        AttentionPackage attention_package;
        RuntimeOptions hybrid_options;
        hybrid_options.hybrid_mode = HybridMode::HybridExperts;
        RuntimeOptions cpu_options;
        cpu_options.hybrid_mode = HybridMode::CpuOnly;
        auto hybrid_model = runtime.load_model(attention_package.path(), hybrid_options);
        auto cpu_model = runtime.load_model(attention_package.path(), cpu_options);
        check(static_cast<bool>(hybrid_model));
        check(static_cast<bool>(cpu_model));

        SchedulerOptions pipeline_options;
        pipeline_options.worker_count = 4;
        auto pipeline_scheduler = runtime.create_scheduler(pipeline_options);
        check(static_cast<bool>(pipeline_scheduler));
        std::vector<SessionPtr> hybrid_sessions;
        std::vector<SessionPtr> cpu_sessions;
        for (uint32_t index = 0; index < 4; ++index)
        {
            auto hybrid_session = runtime.create_session(hybrid_model.value());
            auto cpu_session = runtime.create_session(cpu_model.value());
            check(static_cast<bool>(hybrid_session));
            check(static_cast<bool>(cpu_session));
            hybrid_sessions.push_back(hybrid_session.value());
            cpu_sessions.push_back(cpu_session.value());
        }

        constexpr uint32_t pipeline_rounds = 6;
        for (uint32_t round = 0; round < pipeline_rounds; ++round)
        {
            std::vector<DecodeBatchRequest> requests;
            for (uint32_t session_index = 0; session_index < hybrid_sessions.size(); ++session_index)
            {
                requests.push_back({
                    hybrid_sessions[session_index],
                    static_cast<int32_t>((round + session_index) % 2),
                });
            }
            std::vector<Result<DecodeResult>> pipeline_results = pipeline_scheduler.value()->submit_decode(std::move(requests)).get();
            check(static_cast<bool>(pipeline_results.size() == hybrid_sessions.size()));
            for (uint32_t session_index = 0; session_index < hybrid_sessions.size(); ++session_index)
            {
                auto cpu_result = cpu_sessions[session_index]->decode(static_cast<int32_t>((round + session_index) % 2));
                check(static_cast<bool>(pipeline_results[session_index]));
                check(static_cast<bool>(cpu_result));
                check(static_cast<bool>(pipeline_results[session_index].value().sequence_length == round + 1));
                check(static_cast<bool>(pipeline_results[session_index].value().logits.values.size() == cpu_result.value().logits.values.size()));
                for (size_t logit = 0; logit < cpu_result.value().logits.values.size(); ++logit)
                {
                    check_near(pipeline_results[session_index].value().logits.values[logit], cpu_result.value().logits.values[logit], 1e-4f);
                }
            }
        }
        for (const SessionPtr& session : hybrid_sessions)
        {
            const SessionStatistics session_statistics = session->statistics();
            check(static_cast<bool>(session_statistics.vulkan_attention_blocks == pipeline_rounds));
            check(static_cast<bool>(session_statistics.vulkan_compute_submissions == pipeline_rounds * 2));
            check(static_cast<bool>(session_statistics.vulkan_staging_slot_acquisitions == pipeline_rounds * 2));
        }
        const SchedulerStatistics pipeline_statistics = pipeline_scheduler.value()->statistics();
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
    for (size_t index = 0; index < single_batch.value().logits.values.size(); ++index)
    {
        check_near(chunked.value().logits.values[index], single_batch.value().logits.values[index], 1e-5f);
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
    const float expert_zero_weight = std::exp(normalized_input) / (std::exp(normalized_input) + 1.0f);
    const float expert_one_weight = 1.0f - expert_zero_weight;
    const float hidden_x = 1.0f + expert_zero_weight * normalized_input;
    const float hidden_y = expert_one_weight * normalized_input;
    const float final_scale = std::sqrt((hidden_x * hidden_x + hidden_y * hidden_y) / 2.0f + 1e-5f);

    check_near(result.value().logits.values[0], hidden_x / final_scale, 1e-5f);
    check_near(result.value().logits.values[1], hidden_y / final_scale, 1e-5f);
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 2));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 2));
    check(static_cast<bool>(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({1, 1, 0})));
    if (has_flag(runtime.capabilities().flags, RuntimeCapabilityOpenmpExperts))
    {
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
    check(static_cast<bool>(has_flag(model.value()->descriptor().layers[0].flags, LayerDescriptorAttention)));

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
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_logical_bytes == float32_session.value()->statistics().kv_cache_logical_bytes / 2));
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_allocated_bytes == float32_session.value()->statistics().kv_cache_allocated_bytes / 2));

    for (uint32_t index = 0; index < 16; ++index)
        check(static_cast<bool>(bfloat16_session.value()->decode(static_cast<int32_t>(index % 2))));
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_allocated_bytes <= bfloat16_session.value()->statistics().kv_cache_logical_bytes * 16));
}

void test_attention_graph_without_bias_or_sink()
{
    AttentionPackage package(false, 0, false, false);
    std::ifstream manifest_stream(package.path() / "config.json");
    const std::string manifest_json{std::istreambuf_iterator<char>(manifest_stream), std::istreambuf_iterator<char>()};
    ModelPackage model_package;
    model_package.root = package.path();
    model_package.manifest.model_type = "test_moe";
    model_package.manifest.raw_json = manifest_json;
    FixtureModelAdapter adapter;
    auto descriptor = adapter.parse_model(model_package);
    check(static_cast<bool>(descriptor));
    auto mapping = adapter.map_weights(model_package, descriptor.value());
    check(static_cast<bool>(mapping));
    descriptor.value().layers[0].flags &= ~(LayerDescriptorAttention | LayerDescriptorMoe);
    ModelCompiler compiler;
    auto graph_driven_model = compiler.compile(std::move(descriptor).value(), std::move(mapping).value(), HybridMode::CpuOnly);
    if (!graph_driven_model)
    {
        throw std::runtime_error("graph-driven model compilation failed: " + graph_driven_model.error().message);
    }
    check(static_cast<bool>(has_flag(graph_driven_model.value().layers[0].flags, CompiledLayerAttention)));
    CpuBatch context_hidden(3, 2);
    context_hidden.row(0)[0] = 1.0f;
    context_hidden.row(0)[1] = 0.0f;
    context_hidden.row(1)[0] = 0.0f;
    context_hidden.row(1)[1] = 1.0f;
    context_hidden.row(2)[0] = 0.5f;
    context_hidden.row(2)[1] = -0.5f;
    CpuLayerCache full_cache;
    CpuLayerCache context_cache;
    CpuAttentionExecutionScratch full_scratch;
    CpuAttentionExecutionScratch context_scratch;
    CpuBatch full_output;
    const CompiledModel& compiled = graph_driven_model.value();
    execute_attention_block_into(
        compiled.weights,
        compiled.layers.front().attention,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        7,
        full_cache,
        full_scratch,
        context_hidden,
        full_output);
    append_attention_context_into(
        compiled.weights,
        compiled.layers.front().attention,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        7,
        context_cache,
        context_scratch,
        context_hidden);
    check(full_cache.start_position == context_cache.start_position);
    check(full_cache.token_count == context_cache.token_count);
    check(full_cache.first_slot == context_cache.first_slot);
    check(full_cache.keys == context_cache.keys);
    check(full_cache.values == context_cache.values);

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
    const CompiledLayerPlan& cpu_plan = cpu_model.value()->execution_plan()[0];
    check(static_cast<bool>(cpu_plan.nodes.size() == expected_nodes.size()));
    for (const CompiledNodePlan& node : cpu_plan.nodes)
        check(static_cast<bool>(node.backend == ExecutionBackend::Cpu));

    auto cpu_session = runtime.create_session(cpu_model.value());
    check(static_cast<bool>(cpu_session));
    const std::vector<int32_t> prompt = {0, 1, 0};
    auto cpu_prefill = cpu_session.value()->prefill(prompt);
    check(static_cast<bool>(cpu_prefill));

    if (has_flag(runtime.capabilities().flags, RuntimeCapabilityVulkanAttention))
    {
        RuntimeOptions hybrid_options;
        hybrid_options.hybrid_mode = HybridMode::HybridExperts;
        auto hybrid_model = runtime.load_model(package.path(), hybrid_options);
        check(static_cast<bool>(hybrid_model));
        const CompiledLayerPlan& hybrid_plan = hybrid_model.value()->execution_plan()[0];
        check(static_cast<bool>(hybrid_plan.nodes.size() == expected_nodes.size()));
        for (size_t node_index = 0; node_index < hybrid_plan.nodes.size(); ++node_index)
        {
            check(static_cast<bool>(hybrid_plan.nodes[node_index].backend == (node_index < 5 ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu)));
        }
        const ExecutionGraph& hybrid_graph = hybrid_model.value()->execution_graph();
        size_t vulkan_attention_nodes = 0;
        size_t vulkan_lm_head_nodes = 0;
        for (const ExecutionNode& node : hybrid_graph.nodes)
        {
            if (node.type == ExecutionNodeType::Attention && node.backend == ExecutionBackend::Vulkan)
            {
                ++vulkan_attention_nodes;
            }
            if (node.type == ExecutionNodeType::LmHead && node.backend == ExecutionBackend::Vulkan)
            {
                ++vulkan_lm_head_nodes;
            }
            if (node.type == ExecutionNodeType::Expert || node.type == ExecutionNodeType::ExpertGroup)
                check(static_cast<bool>(node.backend == ExecutionBackend::Cpu));
        }
        check(static_cast<bool>(vulkan_attention_nodes == 1));
        check(static_cast<bool>(vulkan_lm_head_nodes == 1));
        auto hybrid_session = runtime.create_session(hybrid_model.value());
        check(static_cast<bool>(hybrid_session));
        auto hybrid_prefill = hybrid_session.value()->prefill(prompt);
        check(static_cast<bool>(hybrid_prefill));
        check(static_cast<bool>(hybrid_session.value()->statistics().vulkan_attention_blocks == 1));
        check(static_cast<bool>(hybrid_prefill.value().logits.values.size() == cpu_prefill.value().logits.values.size()));
        for (size_t index = 0; index < cpu_prefill.value().logits.values.size(); ++index)
        {
            check_near(hybrid_prefill.value().logits.values[index], cpu_prefill.value().logits.values[index], 1e-4f);
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
    const MoeGraph& ir_graph = model.value()->ir().graph;
    check(static_cast<bool>(ir_graph.validate()));
    check(static_cast<bool>(ir_graph.nodes.size() == 6));
    check(static_cast<bool>(ir_graph.nodes[0].operation == MoeIROperator::TokenEmbedding));
    check(static_cast<bool>(ir_graph.nodes[2].operation == MoeIROperator::ExpertGroup));
    check(static_cast<bool>(ir_graph.nodes[2].quantization.storage_dtype == DType::Float32));
    check(static_cast<bool>(ir_graph.nodes[2].experts.expert_count == 2));
    MoeIR graph_only_ir = model.value()->ir();
    graph_only_ir.layers.clear();
    auto graph_only_status = normalize_moe_ir(graph_only_ir);
    check(static_cast<bool>(graph_only_status));
    check(static_cast<bool>(graph_only_ir.layers.size() == 1));
    check(static_cast<bool>(graph_only_ir.layers[0].nodes.size() == 5));
    MoeIR inconsistent_ir = model.value()->ir();
    inconsistent_ir.graph.nodes[2].experts.top_k = 2;
    auto inconsistent_status = normalize_moe_ir(inconsistent_ir);
    check(static_cast<bool>(!inconsistent_status));
    check(static_cast<bool>(inconsistent_status.error().code == ErrorCode::InvalidModel));

    const ExecutionGraph& graph = model.value()->execution_graph();
    check(static_cast<bool>(graph.validate()));
    check(static_cast<bool>(graph.nodes.size() == 7));
    check(static_cast<bool>(!graph.tensors.empty()));
    const std::vector<ExecutionNodeType> expected_types = {
        ExecutionNodeType::TokenEmbedding,
        ExecutionNodeType::Router,
        ExecutionNodeType::ExpertDispatch,
        ExecutionNodeType::ExpertGroup,
        ExecutionNodeType::Combine,
        ExecutionNodeType::FinalNorm,
        ExecutionNodeType::LmHead,
    };
    for (size_t index = 0; index < expected_types.size(); ++index)
    {
        check(static_cast<bool>(graph.nodes[index].id == index));
        check(static_cast<bool>(graph.nodes[index].type == expected_types[index]));
        check(static_cast<bool>(graph.nodes[index].backend == ExecutionBackend::Cpu));
    }
    check(static_cast<bool>(graph.nodes[3].flags == 0));
    check(static_cast<bool>(graph.nodes[3].expert_id == invalid_execution_expert_id));
    check(static_cast<bool>(graph.nodes[4].dependencies.size() == 1));
    check(static_cast<bool>(graph.nodes[4].dependencies[0] == 3));

    const ExecutionSchedule& compiled_schedule = model.value()->execution_schedule();
    check(static_cast<bool>(compiled_schedule.cpu_parallelism == runtime.capabilities().openmp_thread_count));
    check(static_cast<bool>(compiled_schedule.waves.size() == 7));
    check(static_cast<bool>(compiled_schedule.waves[3].nodes.size() == 1));
    check(static_cast<bool>(graph.find(compiled_schedule.waves[3].nodes[0])->type == ExecutionNodeType::ExpertGroup));
    check(static_cast<bool>(compiled_schedule.waves[3].cpu_nodes.size() == 1));
    check(static_cast<bool>(compiled_schedule.waves[3].vulkan_nodes.empty()));

    const ExpertStoreStatistics initial_experts = model.value()->expert_store().statistics();
    check(static_cast<bool>(initial_experts.expert_count == 2));
    check(static_cast<bool>(initial_experts.resident_experts == 2));
    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));
    const std::array<int32_t, 1> prompt = {0};
    auto prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill));
    const SessionStatistics session_statistics = session.value()->statistics();
    check(static_cast<bool>(session_statistics.expert_batch_weight_bytes > 0));
    check(static_cast<bool>(session_statistics.expert_route_weight_bytes >= session_statistics.expert_batch_weight_bytes));
    const ExpertStoreStatistics used_experts = model.value()->expert_store().statistics();
    check(static_cast<bool>(used_experts.dispatch_count > 0));
    check(static_cast<bool>(used_experts.token_count > 0));
    const ExpertHotsetEstimate empty_hotset = model.value()->expert_store().estimate_hotset(0);
    check(static_cast<bool>(empty_hotset.active_expert_count > 0));
    check(static_cast<bool>(empty_hotset.resident_expert_count == 0));
    check(static_cast<bool>(empty_hotset.covered_batch_weight_bytes == 0));
    const ExpertHotsetEstimate full_hotset = model.value()->expert_store().estimate_hotset(used_experts.registered_weight_bytes);
    check(static_cast<bool>(full_hotset.resident_expert_count == full_hotset.active_expert_count));
    check(static_cast<bool>(full_hotset.covered_batch_weight_bytes == full_hotset.requested_batch_weight_bytes));
    check(static_cast<bool>(full_hotset.covered_route_weight_bytes == full_hotset.requested_route_weight_bytes));
    const MemoryManagerStatistics memory = session.value()->memory_statistics();
    check(static_cast<bool>(memory.registered_tensors == graph.tensors.size()));
    check(static_cast<bool>(memory.tensor_uses > 0));

    MoeScheduler scheduler;
    auto rescheduled = scheduler.schedule(graph);
    check(static_cast<bool>(rescheduled));
    check(static_cast<bool>(rescheduled.value().waves.size() == compiled_schedule.waves.size()));

    ExecutionGraph hybrid_graph = graph;
    hybrid_graph.nodes.back().backend_mask |= ExecutionBackendVulkan;
    RuntimeSchedulingOptions scheduling_options;
    scheduling_options.available_backends = ExecutionBackendCpu | ExecutionBackendVulkan;
    scheduling_options.vulkan_queue_count = 1;
    RuntimeScheduler runtime_scheduler;
    auto hybrid_schedule = runtime_scheduler.compile(std::move(hybrid_graph), scheduling_options);
    check(static_cast<bool>(hybrid_schedule));
    check(static_cast<bool>(hybrid_schedule.value().graph.nodes.back().backend == ExecutionBackend::Vulkan));
    check(static_cast<bool>(!hybrid_schedule.value().graph.events.empty()));
    check(static_cast<bool>(hybrid_schedule.value().schedule.events.size() == hybrid_schedule.value().graph.events.size()));

    ExecutionGraph cyclic;
    ExecutionNode cyclic_router;
    cyclic_router.id = 0;
    cyclic_router.type = ExecutionNodeType::Router;
    cyclic_router.backend = ExecutionBackend::Cpu;
    cyclic_router.backend_mask = ExecutionBackendCpu;
    cyclic_router.layer_id = 0;
    cyclic_router.name = "router";
    cyclic_router.dependencies = {1};
    ExecutionNode cyclic_combine;
    cyclic_combine.id = 1;
    cyclic_combine.type = ExecutionNodeType::Combine;
    cyclic_combine.backend = ExecutionBackend::Cpu;
    cyclic_combine.backend_mask = ExecutionBackendCpu;
    cyclic_combine.layer_id = 0;
    cyclic_combine.name = "combine";
    cyclic_combine.dependencies = {0};
    cyclic.nodes = {
        std::move(cyclic_router),
        std::move(cyclic_combine),
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
    auto weighted = dispatcher.dispatch(weighted_logits, 1, options);
    check(static_cast<bool>(weighted));
    check(static_cast<bool>(weighted.value().assignment_count == 2));
    check(static_cast<bool>(weighted.value().batches.size() == 2));
    check(weighted.value().batches[0].routes[0].rank == 0);
    check(weighted.value().batches[1].routes[0].rank == 1);
    check_near(weighted.value().batches[0].routes[0].weight + weighted.value().batches[1].routes[0].weight, 1.0f, 1e-6f);
    ExpertDispatchPlan reusable;
    auto dispatched_into = dispatcher.dispatch_into(weighted_logits, 1, options, reusable);
    check(static_cast<bool>(dispatched_into));
    check(static_cast<bool>(reusable.assignment_count == weighted.value().assignment_count));
    check(static_cast<bool>(reusable.batches.size() == weighted.value().batches.size()));
    for (size_t index = 0; index < reusable.batches.size(); ++index)
    {
        check(static_cast<bool>(reusable.batches[index].expert_id == weighted.value().batches[index].expert_id));
        check(reusable.batches[index].routes.front().rank == weighted.value().batches[index].routes.front().rank);
        check_near(reusable.batches[index].routes.front().weight, weighted.value().batches[index].routes.front().weight, 1e-6f);
    }
    const ExpertBatch* reused_batches = reusable.batches.data();
    const ExpertRoute* reused_first_route = reusable.batches.front().routes.data();
    const std::vector<float> next_logits = {0.0f, 1.0f, 2.0f};
    dispatched_into = dispatcher.dispatch_into(next_logits, 1, options, reusable);
    check(static_cast<bool>(dispatched_into));
    check(static_cast<bool>(reusable.batches.data() == reused_batches));
    check(static_cast<bool>(reusable.batches.front().routes.data() == reused_first_route));

    const std::vector<float> invalid_logits = {1.0f, 2.0f};
    auto invalid = dispatcher.dispatch(invalid_logits, 1, options);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));

    const auto check_dispatch_into_matches = [&](std::span<const float> test_logits, const ExpertDispatchOptions& test_options) {
        auto expected = dispatcher.dispatch(test_logits, 1, test_options);
        check(static_cast<bool>(expected));
        ExpertDispatchPlan actual;
        auto status = dispatcher.dispatch_into(test_logits, 1, test_options, actual);
        check(static_cast<bool>(status));
        check(actual.assignment_count == expected.value().assignment_count);
        check(actual.batches.size() == expected.value().batches.size());
        for (size_t batch_index = 0; batch_index < actual.batches.size(); ++batch_index)
        {
            check(actual.batches[batch_index].expert_id == expected.value().batches[batch_index].expert_id);
            check(actual.batches[batch_index].routes.size() == expected.value().batches[batch_index].routes.size());
            for (size_t route_index = 0; route_index < actual.batches[batch_index].routes.size(); ++route_index)
            {
                check(actual.batches[batch_index].routes[route_index].token_index == expected.value().batches[batch_index].routes[route_index].token_index);
                check(actual.batches[batch_index].routes[route_index].rank == expected.value().batches[batch_index].routes[route_index].rank);
                check_near(actual.batches[batch_index].routes[route_index].weight, expected.value().batches[batch_index].routes[route_index].weight, 1e-6f);
            }
        }
    };
    ExpertDispatchOptions sigmoid_options;
    sigmoid_options.expert_count = 4;
    sigmoid_options.top_k = 2;
    sigmoid_options.score_function = RouterScoreFunction::Sigmoid;
    sigmoid_options.normalization = RouterNormalization::None;
    sigmoid_options.routed_scaling_factor = 2.0f;
    sigmoid_options.flags = 0;
    const std::array<float, 4> tied_sigmoid_logits = {0.0f, 0.0f, -1.0f, -1.0f};
    check_dispatch_into_matches(tied_sigmoid_logits, sigmoid_options);

    ExpertDispatchOptions wide_options;
    wide_options.expert_count = 20;
    std::array<float, 20> wide_logits;
    for (size_t index = 0; index < wide_logits.size(); ++index)
        wide_logits[index] = static_cast<float>(static_cast<int>(index % 7) - 3);
    wide_options.top_k = 16;
    check_dispatch_into_matches(wide_logits, wide_options);
    wide_options.top_k = 17;
    check_dispatch_into_matches(wide_logits, wide_options);

    ExpertDispatchOptions duplicate_options;
    duplicate_options.expert_count = 4;
    duplicate_options.top_k = 2;
    const std::array<uint32_t, 2> duplicate_experts = {1, 1};
    duplicate_options.explicit_expert_ids = duplicate_experts;
    const std::array<float, 4> duplicate_logits = {0.0f, 1.0f, 2.0f, 3.0f};
    check_dispatch_into_matches(duplicate_logits, duplicate_options);
}

void test_deepseek_router_and_hyper_connection_kernels()
{
    ExpertDispatcher dispatcher;
    ExpertDispatchOptions options;
    options.expert_count = 4;
    options.top_k = 2;
    options.score_function = RouterScoreFunction::SqrtSoftplus;
    options.routed_scaling_factor = 1.5f;
    const std::array<float, 4> selection_bias = {0.0f, 100.0f, 0.0f, 0.0f};
    options.selection_bias = selection_bias;
    const std::array<float, 4> logits = {4.0f, -4.0f, 3.0f, 2.0f};
    auto routed = dispatcher.dispatch(logits, 1, options);
    check(static_cast<bool>(routed));
    check(routed.value().batches.size() == 2);
    check(routed.value().batches[0].expert_id == 0);
    check(routed.value().batches[1].expert_id == 1);
    check_near(routed.value().batches[0].routes[0].weight + routed.value().batches[1].routes[0].weight, 1.5f, 1e-5f);
    ExpertDispatchPlan reusable;
    auto routed_into = dispatcher.dispatch_into(logits, 1, options, reusable);
    check(static_cast<bool>(routed_into));
    check(reusable.assignment_count == routed.value().assignment_count);
    check(reusable.batches.size() == routed.value().batches.size());
    for (size_t index = 0; index < reusable.batches.size(); ++index)
    {
        check(reusable.batches[index].expert_id == routed.value().batches[index].expert_id);
        check_near(reusable.batches[index].routes[0].weight, routed.value().batches[index].routes[0].weight, 1e-6f);
    }
    const ExpertBatch* reusable_batches = reusable.batches.data();
    const ExpertRoute* reusable_route = reusable.batches.front().routes.data();
    routed_into = dispatcher.dispatch_into(logits, 1, options, reusable);
    check(static_cast<bool>(routed_into));
    check(reusable.batches.data() == reusable_batches);
    check(reusable.batches.front().routes.data() == reusable_route);

    const std::array<uint32_t, 2> explicit_experts = {3, 2};
    options.selection_bias = {};
    options.explicit_expert_ids = explicit_experts;
    routed = dispatcher.dispatch(logits, 1, options);
    check(static_cast<bool>(routed));
    check(routed.value().batches[0].expert_id == 2);
    check(routed.value().batches[1].expert_id == 3);
    routed_into = dispatcher.dispatch_into(logits, 1, options, reusable);
    check(static_cast<bool>(routed_into));
    check(reusable.batches[0].expert_id == 2);
    check(reusable.batches[1].expert_id == 3);

    CpuBatch hyper_input(1, 2);
    hyper_input.row(0)[0] = 2.0f;
    hyper_input.row(0)[1] = 4.0f;
    TensorData function;
    function.dtype = DType::Float32;
    function.shape = {8, 2};
    function.float32_data.resize(16, 0.0f);
    TensorData base;
    base.dtype = DType::Float32;
    base.shape = {8};
    base.float32_data.resize(8, 0.0f);
    TensorData scale;
    scale.dtype = DType::Float32;
    scale.shape = {3};
    scale.float32_data.resize(3, 0.0f);
    auto mixed = hyper_connection_pre(hyper_input, function, scale, base, 2, 2, 1e-6f, 1e-6f);
    check(static_cast<bool>(mixed));
    check_near(mixed.value().reduced.row(0)[0], 3.000006f, 1e-4f);
    CpuBatch branch(1, 1);
    branch.row(0)[0] = 10.0f;
    auto connected = hyper_connection_post(branch, hyper_input, mixed.value(), 2);
    check(static_cast<bool>(connected));
    check_near(connected.value().row(0)[0], 13.0f, 1e-3f);
    check_near(connected.value().row(0)[1], 13.0f, 1e-3f);
    CpuHyperConnectionMix directed_mix;
    directed_mix.post = {0.0f, 0.0f};
    directed_mix.combine = {1.0f, 2.0f, 3.0f, 4.0f};
    branch.row(0)[0] = 0.0f;
    connected = hyper_connection_post(branch, hyper_input, directed_mix, 2);
    check(static_cast<bool>(connected));
    check_near(connected.value().row(0)[0], 14.0f, 1e-5f);
    check_near(connected.value().row(0)[1], 20.0f, 1e-5f);

    for (float value : std::array<float, 7>{-448.0f, -1.5f, -0.001f, 0.0f, 0.5f, 6.0f, 448.0f})
    {
        const float round_trip = float8_e4m3_to_float(float_to_float8_e4m3(value));
        check(std::isfinite(round_trip));
        check(std::signbit(round_trip) == std::signbit(value) || value == 0.0f);
    }
    std::array<uint8_t, 256> float8_weights = {};
    std::array<float, 256> float8_input = {};
    for (size_t index = 0; index < float8_weights.size(); ++index)
    {
        uint8_t encoded = static_cast<uint8_t>(index);
        if ((encoded & UINT8_C(0x7f)) == UINT8_C(0x7f))
            encoded ^= UINT8_C(1);
        float8_weights[index] = encoded;
        float8_input[index] = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.015625f;
    }
    const std::array<float, 2> float8_scales = {0.5f, 2.0f};
    float float8_reference = 0.0f;
    for (size_t block = 0; block < float8_scales.size(); ++block)
    {
        float partial = 0.0f;
        for (size_t index = block * 128; index < (block + 1) * 128; ++index)
            partial += float8_e4m3_to_float(float8_weights[index]) * float8_input[index];
        float8_reference += partial * float8_scales[block];
    }
    check_near(float8_e4m3_block_dot(float8_weights.data(), float8_scales.data(), float8_input.data(), 256, 128), float8_reference, 1e-3f);
    std::array<uint8_t, 4 * 256> float8_row_weights = {};
    std::array<float, 4> float8_row_reference = {};
    std::array<float, 4> float8_row_output = {};
    for (size_t row = 0; row < float8_row_reference.size(); ++row)
    {
        for (size_t index = 0; index < float8_weights.size(); ++index)
        {
            uint8_t encoded = static_cast<uint8_t>(index + row * 13);
            if ((encoded & UINT8_C(0x7f)) == UINT8_C(0x7f))
                encoded ^= UINT8_C(1);
            float8_row_weights[row * float8_weights.size() + index] = encoded;
            float8_row_reference[row] += float8_e4m3_to_float(encoded) * float8_input[index] * float8_scales[index / 128];
        }
    }
    float8_e4m3_block_dot_rows4(
        float8_row_weights.data(),
        256,
        float8_scales.data(),
        float8_input.data(),
        256,
        128,
        4,
        float8_row_output.data());
    for (size_t row = 0; row < float8_row_output.size(); ++row)
        check_near(float8_row_output[row], float8_row_reference[row], 1e-3f);
    check(static_cast<bool>(std::string(float8_kernel_name()).size() > 0));
}

static ModelPackage deepseek_v4_package(const std::string& dspark_fields)
{
    ModelPackage package;
    package.manifest.model_type = "deepseek_v4";
    package.manifest.raw_json = R"({
        "vocab_size": 129280,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 4,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "sliding_window": 128,
        "max_position_embeddings": 1048576,
        "original_max_position_embeddings": 65536,
        "num_hash_layers": 3,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "compress_ratios": [0, 0, 4, 128],
)" + dspark_fields + R"(
        "expert_dtype": "fp4",
        "scoring_func": "sqrtsoftplus",
        "quant_method": "fp8",
        "fmt": "e4m3",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
        "rms_norm_eps": 0.000001,
        "hc_eps": 0.000001,
        "swiglu_limit": 10.0,
        "routed_scaling_factor": 1.5
    })";
    return package;
}

void test_deepseek_v4_descriptors()
{
    DeepSeekV4ModelAdapter adapter;
    auto flash = adapter.parse_model(deepseek_v4_package(""));
    check(static_cast<bool>(flash));
    check(flash.value().speculative_layer_count == 0);
    check(flash.value().speculative_target_layer_ids.empty());

    auto parsed = adapter.parse_model(deepseek_v4_package(R"(
        "dspark_target_layer_ids": [1, 2, 3],
        "dspark_block_size": 5,
        "dspark_noise_token_id": 127,
        "dspark_markov_rank": 256,
)"));
    check(static_cast<bool>(parsed));
    const MoeIR& descriptor = parsed.value();
    check(descriptor.model_type == "deepseek_v4");
    check(descriptor.hyper_connection_multiplier == 4);
    check(descriptor.hash_routing_layer_count == 3);
    check(descriptor.speculative_kind == SpeculativeModelKind::DSpark);
    check(descriptor.speculative_layer_count == 3);
    check(descriptor.speculative_block_size == 5);
    check(descriptor.speculative_noise_token_id == 127);
    check(descriptor.speculative_markov_rank == 256);
    check(descriptor.layers[2].attention.kind == AttentionKind::MultiHeadLatent);
    check(descriptor.layers[2].attention.compression_ratio == 4);
    check(descriptor.layers[3].attention.compression_ratio == 128);
    check(descriptor.layers[0].ffn.moe.score_function == RouterScoreFunction::SqrtSoftplus);
    check(descriptor.layers[0].ffn.moe.shared_expert_count == 1);
    check(has_flag(descriptor.layers[0].ffn.moe.flags, MoeDescriptorSharedExpert));
    RuntimeOptions options;
    auto memory = plan_model_memory(descriptor, options, UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(memory));
    check(memory.value().selected_mode == ExpertMemoryMode::OnDemand);
    check(memory.value().estimated_dense_bytes < UINT64_C(10) * 1024 * 1024 * 1024);

    auto partial_dspark = adapter.parse_model(deepseek_v4_package(R"(
        "dspark_block_size": 5,
)"));
    check(!partial_dspark);
}

static ModelPackage qwen3_5_moe_package()
{
    ModelPackage package;
    package.manifest.model_type = "qwen3_5_moe";
    package.manifest.raw_json = R"({
        "vocab_size": 128,
        "hidden_size": 16,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "num_hidden_layers": 4,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 2,
        "max_position_embeddings": 128,
        "rms_norm_eps": 0.000001,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ],
        "hidden_act": "silu",
        "dtype": "bfloat16",
        "mamba_ssm_dtype": "float32",
        "attention_bias": false,
        "attn_output_gate": true
    })";
    return package;
}

static uint64_t qwen_test_fnv1a64(const std::string& bytes)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char value : bytes)
    {
        hash ^= value;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void write_qwen_mxfp4_test_artifact(
    ModelPackage& package,
    const std::filesystem::path& root)
{
    constexpr uint32_t layer_count = 4;
    constexpr uint32_t expert_count = 4;
    constexpr uint32_t hidden_size = 32;
    constexpr uint32_t intermediate_size = 32;
    package.root = root;
    package.manifest.raw_json = std::regex_replace(
        package.manifest.raw_json,
        std::regex(R"("hidden_size"\s*:\s*16)"),
        R"("hidden_size": 32)");
    package.manifest.raw_json = std::regex_replace(
        package.manifest.raw_json,
        std::regex(R"("moe_intermediate_size"\s*:\s*4)"),
        R"("moe_intermediate_size": 32)");
    package.manifest.raw_json = std::regex_replace(
        package.manifest.raw_json,
        std::regex(R"("shared_expert_intermediate_size"\s*:\s*4)"),
        R"("shared_expert_intermediate_size": 32)");
    const std::string index_json = "{}\n";
    {
        std::ofstream config(root / "config.json", std::ios::binary);
        config << package.manifest.raw_json;
    }
    {
        std::ofstream index(
            root / "model.safetensors.index.json",
            std::ios::binary);
        index << index_json;
    }

    std::ostringstream identity;
    identity << "__ncnn_moe_qwen3_6_mxfp4__.identity.v3."
             << layer_count << ".1."
             << expert_count << '.'
             << hidden_size << '.'
             << intermediate_size << '.'
             << std::hex << std::setfill('0')
             << std::setw(16)
             << qwen_test_fnv1a64(package.manifest.raw_json)
             << '.'
             << std::setw(16)
             << qwen_test_fnv1a64(index_json);

    std::ostringstream header;
    header << R"({"__metadata__":{"format":"ncnn-moe-qwen3.6-mxfp4-v3"})";
    uint64_t data_offset = 0;
    const auto add_tensor = [&](const std::string& name, const std::vector<uint32_t>& shape) {
        uint64_t byte_count = 1;
        header << ",\"" << name << "\":{\"dtype\":\"U8\",\"shape\":[";
        for (size_t index = 0; index < shape.size(); ++index)
        {
            if (index != 0)
                header << ',';
            header << shape[index];
            byte_count *= shape[index];
        }
        header << "],\"data_offsets\":["
               << data_offset << ','
               << data_offset + byte_count << "]}";
        data_offset += byte_count;
    };
    add_tensor(identity.str(), {0});
    for (uint32_t layer_id = 0; layer_id < layer_count; ++layer_id)
    {
        const std::string prefix = "__ncnn_moe_qwen3_6_mxfp4__.layers."
                                   + std::to_string(layer_id)
                                   + ".experts.";
        add_tensor(
            prefix + "gate_up.blocks",
            {expert_count, intermediate_size * 2, hidden_size / 32, 16});
        add_tensor(
            prefix + "gate_up.scales",
            {expert_count, intermediate_size * 2, hidden_size / 32});
        add_tensor(
            prefix + "down.blocks",
            {expert_count, hidden_size, intermediate_size / 32, 16});
        add_tensor(
            prefix + "down.scales",
            {expert_count, hidden_size, intermediate_size / 32});
    }
    const std::string mtp_prefix =
        "__ncnn_moe_qwen3_6_mxfp4__.mtp.layers.0.experts.";
    add_tensor(
        mtp_prefix + "gate_up.blocks",
        {expert_count, intermediate_size * 2, hidden_size / 32, 16});
    add_tensor(
        mtp_prefix + "gate_up.scales",
        {expert_count, intermediate_size * 2, hidden_size / 32});
    add_tensor(
        mtp_prefix + "down.blocks",
        {expert_count, hidden_size, intermediate_size / 32, 16});
    add_tensor(
        mtp_prefix + "down.scales",
        {expert_count, hidden_size, intermediate_size / 32});
    header << '}';
    std::string encoded_header = header.str();
    encoded_header.append(
        (8 - encoded_header.size() % 8) % 8,
        ' ');
    std::ofstream artifact(
        root / "ncnn-moe-qwen3.6-mxfp4.safetensors",
        std::ios::binary);
    const uint64_t header_bytes = encoded_header.size();
    artifact.write(
        reinterpret_cast<const char*>(&header_bytes),
        sizeof(header_bytes));
    artifact.write(
        encoded_header.data(),
        static_cast<std::streamsize>(encoded_header.size()));
    std::array<char, 4096> zeros = {};
    for (uint64_t written = 0; written < data_offset;)
    {
        const uint64_t count = std::min<uint64_t>(
            zeros.size(),
            data_offset - written);
        artifact.write(
            zeros.data(),
            static_cast<std::streamsize>(count));
        written += count;
    }
}

void test_qwen3_5_moe_descriptors()
{
    Qwen3_5MoeModelAdapter adapter;
    auto parsed = adapter.parse_model(qwen3_5_moe_package());
    check(static_cast<bool>(parsed));
    const MoeIR& descriptor = parsed.value();
    check(descriptor.model_type == "qwen3_5_moe");
    check(descriptor.norm_weight_offset == 1.0f);
    check(descriptor.layer_count == 4);
    check(descriptor.layers[0].attention.kind == AttentionKind::GatedDeltaNet);
    check(descriptor.layers[0].attention.head_count == 2);
    check(descriptor.layers[0].attention.kv_head_count == 1);
    check(descriptor.layers[0].attention.head_dimension == 2);
    check(descriptor.layers[0].attention.value_head_dimension == 2);
    check(descriptor.layers[0].attention.convolution_kernel_size == 2);
    check(descriptor.layers[3].attention.kind == AttentionKind::Standard);
    check(descriptor.layers[3].attention.qk_rope_head_dimension == 2);
    check(has_flag(descriptor.layers[3].attention.flags, AttentionDescriptorQueryKeyNorm));
    check(has_flag(descriptor.layers[3].attention.flags, AttentionDescriptorOutputGate));
    const MoeDescriptor& moe = descriptor.layers[0].ffn.moe;
    check(moe.expert_weight_dtype == DType::BFloat16);
    check(moe.layout == ExpertLayout::PackedGateUpDown);
    check(moe.normalization == RouterNormalization::SelectedExperts);
    check(has_flag(moe.flags, MoeDescriptorSharedExpert));
    check(has_flag(moe.flags, MoeDescriptorSharedExpertGate));
    RuntimeOptions options;
    auto memory = plan_model_memory(
        descriptor,
        options,
        UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(memory));
    check(memory.value().selected_mode == ExpertMemoryMode::Eager);
    check(memory.value().expert_pair_bytes == 384);
    check(memory.value().estimated_expert_bytes == 6144);
    MoeGraphBuilder graph_builder;
    auto graph = graph_builder.build(descriptor);
    check(static_cast<bool>(graph));
    const auto delta_cache = std::find_if(
        graph.value().values.begin(),
        graph.value().values.end(),
        [](const MoeIRValue& value) {
            return value.name == "layers.0.kv_cache";
        });
    const auto full_attention_cache = std::find_if(
        graph.value().values.begin(),
        graph.value().values.end(),
        [](const MoeIRValue& value) {
            return value.name == "layers.3.kv_cache";
        });
    check(delta_cache != graph.value().values.end());
    check(delta_cache->dtype == DType::Float32);
    check(delta_cache->shape == std::vector<uint32_t>({0, 2, 2, 2}));
    check(full_attention_cache != graph.value().values.end());
    check(full_attention_cache->dtype == DType::BFloat16);

    ModelPackage invalid = qwen3_5_moe_package();
    invalid.manifest.raw_json = std::regex_replace(
        invalid.manifest.raw_json,
        std::regex(R"("mamba_ssm_dtype"\s*:\s*"float32")"),
        R"("mamba_ssm_dtype": "bfloat16")");
    check(!adapter.parse_model(invalid));

    ScopedTestDirectory artifact_directory("ncnn_moe_qwen_artifact_test_");
    ModelPackage artifact_package = qwen3_5_moe_package();
    write_qwen_mxfp4_test_artifact(
        artifact_package,
        artifact_directory.path());
    auto artifact_descriptor = adapter.parse_model(artifact_package);
    check(static_cast<bool>(artifact_descriptor));
    check(artifact_descriptor.value().layers[0].ffn.moe.expert_weight_dtype
          == DType::MxFp4);
    check(artifact_descriptor.value().speculative_kind
          == SpeculativeModelKind::Mtp);
    check(artifact_descriptor.value().speculative_layer_count == 1);
    check(artifact_descriptor.value().speculative_block_size == 2);

    {
        std::ofstream changed_index(
            artifact_directory.path() / "model.safetensors.index.json",
            std::ios::binary | std::ios::app);
        changed_index << ' ';
    }
    auto stale_artifact = adapter.parse_model(artifact_package);
    check(!stale_artifact);
    check(stale_artifact.error().code == ErrorCode::InvalidModel);
}

static TensorHandle add_float_tensor(
    WeightTable& weights,
    const std::string& name,
    std::vector<uint32_t> shape,
    std::vector<float> values)
{
    TensorData tensor;
    tensor.dtype = DType::Float32;
    tensor.shape = std::move(shape);
    tensor.float32_data = std::move(values);
    auto added = weights.add(name, std::move(tensor));
    check(static_cast<bool>(added));
    return added.value();
}

void test_gated_delta_net_continuation()
{
    WeightTable weights;
    AttentionBlockPlan plan;
    plan.head_count = 1;
    plan.kv_head_count = 1;
    plan.head_dimension = 2;
    plan.value_head_dimension = 2;
    plan.convolution_kernel_size = 2;
    plan.norm_weight_offset = 1.0f;
    plan.flags = AttentionBlockGatedDeltaNet;

    plan.pre_attention_norm_weight = add_float_tensor(weights, "pre_norm", {2}, {0.0f, 0.0f});
    plan.delta_qkv_weight = add_float_tensor(
        weights,
        "qkv",
        {6, 2},
        {
            0.50f,
            -0.25f,
            0.25f,
            0.75f,
            -0.50f,
            0.25f,
            0.75f,
            0.50f,
            0.30f,
            -0.20f,
            -0.40f,
            0.60f,
        });
    plan.delta_z_weight = add_float_tensor(weights, "z", {2, 2}, {0.40f, 0.10f, -0.20f, 0.50f});
    plan.delta_beta_weight = add_float_tensor(weights, "beta", {1, 2}, {0.25f, -0.35f});
    plan.delta_alpha_weight = add_float_tensor(weights, "alpha", {1, 2}, {-0.15f, 0.45f});
    plan.delta_convolution_weight = add_float_tensor(
        weights,
        "conv",
        {6, 1, 2},
        {
            0.20f,
            0.80f,
            -0.10f,
            0.70f,
            0.30f,
            0.60f,
            0.15f,
            0.90f,
            -0.25f,
            0.50f,
            0.40f,
            0.65f,
        });
    plan.delta_time_bias = add_float_tensor(weights, "time_bias", {1}, {0.10f});
    plan.delta_decay_log = add_float_tensor(weights, "decay_log", {1}, {-0.20f});
    plan.delta_norm_weight = add_float_tensor(weights, "delta_norm", {2}, {1.10f, 0.90f});
    plan.output_weight = add_float_tensor(weights, "output", {2, 2}, {0.70f, -0.10f, 0.20f, 0.80f});

    CpuBatch input(2, 2);
    input.row(0)[0] = 0.75f;
    input.row(0)[1] = -0.25f;
    input.row(1)[0] = -0.40f;
    input.row(1)[1] = 0.90f;

    CpuLayerCache prefill_cache;
    CpuGatedDeltaExecutionScratch prefill_scratch;
    CpuBatch prefill_output;
    execute_gated_delta_net_into(
        weights,
        plan,
        1e-6f,
        prefill_cache,
        prefill_scratch,
        input,
        prefill_output);

    CpuLayerCache decode_cache;
    CpuGatedDeltaExecutionScratch decode_scratch;
    CpuBatch decode_output(2, 2);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        CpuBatch token(1, 2);
        std::copy_n(input.row(row), 2, token.row(0));
        CpuBatch token_output;
        execute_gated_delta_net_into(
            weights,
            plan,
            1e-6f,
            decode_cache,
            decode_scratch,
            token,
            token_output);
        std::copy_n(token_output.row(0), 2, decode_output.row(row));
    }

    check(prefill_cache.gated_delta_token_count == 2);
    check(decode_cache.gated_delta_token_count == 2);
    const float expected_output[2][2] = {
        {0.57723981f, -0.45983201f},
        {-0.41938064f, 1.32041629f},
    };
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (size_t column = 0; column < input.columns(); ++column)
        {
            check_near(decode_output.row(row)[column], prefill_output.row(row)[column], 1e-6f);
            check_near(prefill_output.row(row)[column], expected_output[row][column], 1e-5f);
        }
    }
    check(prefill_cache.gated_delta_convolution.size() == decode_cache.gated_delta_convolution.size());
    for (size_t index = 0; index < prefill_cache.gated_delta_convolution.size(); ++index)
        check_near(decode_cache.gated_delta_convolution[index], prefill_cache.gated_delta_convolution[index], 1e-6f);
    check(prefill_cache.gated_delta_recurrent.size() == decode_cache.gated_delta_recurrent.size());
    for (size_t index = 0; index < prefill_cache.gated_delta_recurrent.size(); ++index)
        check_near(decode_cache.gated_delta_recurrent[index], prefill_cache.gated_delta_recurrent[index], 1e-6f);
    const std::array<float, 4> expected_recurrent = {
        -0.03332394f,
        0.04625921f,
        -0.02358976f,
        0.03136346f,
    };
    for (size_t index = 0; index < expected_recurrent.size(); ++index)
        check_near(prefill_cache.gated_delta_recurrent[index], expected_recurrent[index], 1e-5f);

#if NCNN_MOE_WITH_NCNN
    plan.fused_delta_input_operator =
        NcnnLinearOperator::create_fused(
            {
                &weights.at(plan.delta_qkv_weight),
                &weights.at(plan.delta_z_weight),
                &weights.at(plan.delta_beta_weight),
                &weights.at(plan.delta_alpha_weight),
            },
            {nullptr, nullptr, nullptr, nullptr},
            NcnnLinearDevice::Cpu);
    check(static_cast<bool>(plan.fused_delta_input_operator));
    CpuLayerCache fused_cache;
    CpuGatedDeltaExecutionScratch fused_scratch;
    CpuBatch fused_output;
    execute_gated_delta_net_into(
        weights,
        plan,
        1e-6f,
        fused_cache,
        fused_scratch,
        input,
        fused_output);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (size_t column = 0; column < input.columns(); ++column)
        {
            check_near(
                fused_output.row(row)[column],
                prefill_output.row(row)[column],
                1e-5f);
        }
    }
    check(fused_cache.gated_delta_convolution.size()
          == prefill_cache.gated_delta_convolution.size());
    for (size_t index = 0;
         index < fused_cache.gated_delta_convolution.size();
         ++index)
    {
        check_near(
            fused_cache.gated_delta_convolution[index],
            prefill_cache.gated_delta_convolution[index],
            1e-5f);
    }
    check(fused_cache.gated_delta_recurrent.size()
          == prefill_cache.gated_delta_recurrent.size());
    for (size_t index = 0;
         index < fused_cache.gated_delta_recurrent.size();
         ++index)
    {
        check_near(
            fused_cache.gated_delta_recurrent[index],
            prefill_cache.gated_delta_recurrent[index],
            1e-5f);
    }
#endif

    CpuLayerCache first_row_cache;
    CpuGatedDeltaExecutionScratch first_row_scratch;
    CpuBatch first_row_input(1, 2);
    std::copy_n(input.row(0), 2, first_row_input.row(0));
    CpuBatch first_row_output;
    execute_gated_delta_net_into(
        weights,
        plan,
        1e-6f,
        first_row_cache,
        first_row_scratch,
        first_row_input,
        first_row_output);

    std::array<CpuLayerCache, 1> committed_cache;
    begin_state_cache_transaction(committed_cache, 2);
    CpuGatedDeltaExecutionScratch committed_scratch;
    CpuBatch committed_output;
    execute_gated_delta_net_into(
        weights,
        plan,
        1e-6f,
        committed_cache.front(),
        committed_scratch,
        input,
        committed_output);
    auto committed = finish_state_cache_transaction(
        committed_cache,
        1);
    check(static_cast<bool>(committed));
    check(committed_cache.front().gated_delta_token_count == 1);
    check(committed_cache.front().gated_delta_convolution
          == first_row_cache.gated_delta_convolution);
    check(committed_cache.front().gated_delta_recurrent
          == first_row_cache.gated_delta_recurrent);

    std::array<CpuLayerCache, 1> rolled_back_cache;
    begin_state_cache_transaction(rolled_back_cache, 2);
    CpuGatedDeltaExecutionScratch rolled_back_scratch;
    CpuBatch rolled_back_output;
    execute_gated_delta_net_into(
        weights,
        plan,
        1e-6f,
        rolled_back_cache.front(),
        rolled_back_scratch,
        input,
        rolled_back_output);
    auto rolled_back = finish_state_cache_transaction(
        rolled_back_cache,
        0);
    check(static_cast<bool>(rolled_back));
    check(rolled_back_cache.front().gated_delta_token_count == 0);
    check(rolled_back_cache.front().gated_delta_convolution.empty());
    check(rolled_back_cache.front().gated_delta_recurrent.empty());

    std::array<CpuLayerCache, 1> standard_cache;
    standard_cache.front().token_count = 5;
    begin_state_cache_transaction(standard_cache, 4);
    standard_cache.front().token_count = 9;
    record_standard_cache_transaction_rows(
        standard_cache.front(),
        4);
    auto standard_committed = finish_state_cache_transaction(
        standard_cache,
        2);
    check(static_cast<bool>(standard_committed));
    check(standard_cache.front().token_count == 7);
}

static MoeIR gpt_oss_memory_ir(uint32_t layer_count, uint32_t expert_count)
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
    for (LayerDescriptor& layer : ir.layers)
    {
        layer.flags |= LayerDescriptorAttention;
        layer.attention.head_count = 64;
        layer.attention.kv_head_count = 8;
        layer.attention.head_dimension = 64;
        layer.attention.flags |= AttentionDescriptorBias | AttentionDescriptorSinks;
        layer.ffn.moe.expert_count = expert_count;
        layer.ffn.moe.top_k = 4;
        layer.ffn.moe.intermediate_size = 2880;
        layer.ffn.moe.expert_weight_dtype = DType::MxFp4;
        layer.ffn.moe.flags |= MoeDescriptorRouterBias | MoeDescriptorProjectionBias;
    }
    return ir;
}

void test_automatic_expert_memory_planning()
{
    static constexpr uint64_t gibibyte = 1024ull * 1024ull * 1024ull;
    RuntimeOptions options;
    const uint64_t physical_memory = 32 * gibibyte;

    const MoeIR small = gpt_oss_memory_ir(24, 32);
    auto small_plan = plan_model_memory(small, options, physical_memory);
    check(static_cast<bool>(small_plan));
    check(static_cast<bool>(small_plan.value().selected_mode == ExpertMemoryMode::Eager));
    check(static_cast<bool>(!has_flag(small_plan.value().flags, ModelMemoryFileBackedExperts)));
    check(static_cast<bool>(small_plan.value().estimated_expert_bytes < 11 * gibibyte));

    const MoeIR large = gpt_oss_memory_ir(36, 128);
    auto large_plan = plan_model_memory(large, options, physical_memory);
    check(static_cast<bool>(large_plan));
    check(static_cast<bool>(large_plan.value().selected_mode == ExpertMemoryMode::OnDemand));
    check(static_cast<bool>(has_flag(large_plan.value().flags, ModelMemoryFileBackedExperts)));
    check(static_cast<bool>(large_plan.value().host_memory_budget_bytes == 24 * gibibyte));
    check(static_cast<bool>(large_plan.value().estimated_dense_bytes == 4334742144ull));
    check(static_cast<bool>(large_plan.value().expert_pair_bytes == 13219200ull));
    check(static_cast<bool>(large_plan.value().estimated_expert_bytes == 60914073600ull));
    check(static_cast<bool>(large_plan.value().expert_cache_bytes == 20 * gibibyte - large_plan.value().estimated_dense_bytes));
    check(static_cast<bool>(large_plan.value().expert_cache_bytes >= large_plan.value().minimum_active_expert_bytes));

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

    logits.values = {3.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    greedy = session.value()->sample(logits, greedy_options);
    check(static_cast<bool>(greedy));
    check(static_cast<bool>(greedy.value().token_id == 0));

    logits.values = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    greedy = session.value()->sample(logits, greedy_options);
    check(static_cast<bool>(!greedy));
    check(static_cast<bool>(greedy.error().code == ErrorCode::InvalidArgument));

    logits.values = {1.0f, 2.0f, 3.0f};
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
        std::vector<int32_t>{0}, generation_options,
        [&streamed, &session](const StreamToken& token) {
            check(session.value()->statistics().prefill_tokens == 1);
            streamed.push_back(token);
            return true;
        },
        [](int32_t token_id) { return "<" + std::to_string(token_id) + ">"; });
    check(static_cast<bool>(generated));
    check(static_cast<bool>(generated.value().tokens.size() == 3));
    check(static_cast<bool>(streamed.size() == 3));
    check(static_cast<bool>(generated.value().tokens[0].text == "<" + std::to_string(generated.value().tokens[0].token_id) + ">"));
    check(static_cast<bool>(session.value()->sequence_length() == 3));

    auto stopped_session = runtime.create_session(model.value(), session_options);
    check(static_cast<bool>(stopped_session));
    auto stopped = stopped_session.value()->generate(std::vector<int32_t>{0}, generation_options, [](const StreamToken&) { return false; });
    check(static_cast<bool>(stopped));
    check(static_cast<bool>(stopped.value().stopped_by_callback));
    check(static_cast<bool>(stopped.value().tokens.size() == 1));
    check(static_cast<bool>(stopped_session.value()->sequence_length() == 1));
}

void test_model_adapter_scopes()
{
    BuiltinModelAdapter adapter;
    ModelManifest manifest;
    manifest.model_type = "gpt_oss";
    check(adapter.can_load(manifest));

    manifest.model_type = "unsupported_moe";
    check(!adapter.can_load(manifest));

    ModelPackage package;
    package.manifest = manifest;
    auto descriptor = adapter.parse_model(package);
    check(!descriptor);
    check(descriptor.error().code == ErrorCode::UnsupportedModel);

    DeepSeekV4ModelAdapter deepseek_adapter;
    manifest.model_type = "deepseek_v4";
    check(deepseek_adapter.can_load(manifest));
    check(!adapter.can_load(manifest));

    manifest.model_type = "unsupported_moe";
    package.manifest = manifest;
    descriptor = deepseek_adapter.parse_model(package);
    check(!descriptor);
    check(descriptor.error().code == ErrorCode::UnsupportedModel);
}

void test_loader_reports_adapter_and_weight_errors()
{
    {
        TemporaryModelPackage package;
        TestRuntime runtime;
        RuntimeOptions options;
        options.expert_gpu_victim_reuse_probe_interval = 0;
        auto model = runtime.load_model(package.path(), options);
        check(static_cast<bool>(!model));
        check(static_cast<bool>(model.error().code == ErrorCode::InvalidArgument));
    }

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

    {
        TemporaryModelPackage package;
        TestRuntime runtime;
        RuntimeOptions options;
        options.expert_memory_mode = ExpertMemoryMode::Eager;
        options.expert_gpu_victim_cache_bytes = 64 * 1024 * 1024;
        auto model = runtime.load_model(package.path(), options);
        check(static_cast<bool>(!model));
        check(static_cast<bool>(model.error().code == ErrorCode::InvalidArgument));
    }
}

void test_backend_capabilities_and_hybrid_execution()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    auto null_cache_sync = runtime.synchronize_model_caches({});
    check(static_cast<bool>(!null_cache_sync));
    check(static_cast<bool>(null_cache_sync.error().code == ErrorCode::InvalidArgument));
    check(static_cast<bool>(has_flag(runtime.capabilities().flags, RuntimeCapabilityCpuExecution)));
    check(static_cast<bool>(has_flag(runtime.capabilities().flags, RuntimeCapabilityVulkanCpuPrefetch) == has_flag(runtime.capabilities().flags, RuntimeCapabilityVulkanCpuMix)));
    RuntimeOptions invalid_device_options;
    invalid_device_options.vulkan_device_index = static_cast<uint32_t>(runtime.capabilities().vulkan_devices.size());
    auto invalid_device_model = runtime.load_model(package.path(), invalid_device_options);
    check(static_cast<bool>(!invalid_device_model));
    check(static_cast<bool>(invalid_device_model.error().code == ErrorCode::InvalidArgument));
    if (!runtime.capabilities().vulkan_devices.empty())
    {
        RuntimeOptions duplicate_devices;
        duplicate_devices.vulkan_device_indices = {
            0,
            0,
        };
        auto duplicate_device_model = runtime.load_model(package.path(), duplicate_devices);
        check(static_cast<bool>(!duplicate_device_model));
        check(static_cast<bool>(duplicate_device_model.error().code == ErrorCode::InvalidArgument));
    }

    RuntimeOptions automatic_options;
    automatic_options.hybrid_mode = HybridMode::Auto;
    const uint32_t automatic_device_index = runtime.capabilities().selected_vulkan_device_index;
    const bool automatic_uses_vulkan = has_flag(runtime.capabilities().flags, RuntimeCapabilityVulkanCpuMix) && automatic_device_index < runtime.capabilities().vulkan_devices.size()
                                       && runtime.capabilities().vulkan_devices[automatic_device_index].type != VulkanDeviceType::Cpu;
    auto automatic_model = runtime.load_model(package.path(), automatic_options);
    check(static_cast<bool>(automatic_model));
    check(static_cast<bool>(runtime.synchronize_model_caches(automatic_model.value())));
    check(static_cast<bool>(automatic_model.value()->hybrid_mode() == (automatic_uses_vulkan ? HybridMode::HybridExperts : HybridMode::CpuOnly)));
    check(static_cast<bool>(automatic_model.value()->vulkan_device_index() == (automatic_uses_vulkan ? automatic_device_index : automatic_vulkan_device_index)));
    auto automatic_session = runtime.create_session(automatic_model.value());
    check(static_cast<bool>(automatic_session));
    const std::vector<int32_t> packed_prompt = {0, 1, 2, 3};
    auto automatic_prefill = automatic_session.value()->prefill(packed_prompt);
    check(static_cast<bool>(automatic_prefill));
    if (automatic_uses_vulkan)
    {
        RuntimeOptions selected_device_options;
        selected_device_options.hybrid_mode = HybridMode::HybridExperts;
        selected_device_options.vulkan_device_index = runtime.capabilities().selected_vulkan_device_index;
        auto selected_device_model = runtime.load_model(package.path(), selected_device_options);
        check(static_cast<bool>(selected_device_model));
        check(static_cast<bool>(selected_device_model.value()->vulkan_device_index() == selected_device_options.vulkan_device_index));

        if (runtime.capabilities().vulkan_devices.size() >= 2 && runtime.capabilities().vulkan_devices[0].type != VulkanDeviceType::Cpu && runtime.capabilities().vulkan_devices[1].type != VulkanDeviceType::Cpu)
        {
            RuntimeOptions multi_device_options;
            multi_device_options.hybrid_mode = HybridMode::HybridExperts;
            multi_device_options.vulkan_device_indices = {
                0,
                1,
            };
            auto multi_device_model = runtime.load_model(package.path(), multi_device_options);
            check(static_cast<bool>(multi_device_model));
            check(static_cast<bool>(multi_device_model.value()->vulkan_device_indices() == std::vector<uint32_t>({0})));
            check(static_cast<bool>(multi_device_model.value()->execution_plan().front().vulkan_device_index < runtime.capabilities().vulkan_devices.size()));
        }

        const SessionStatistics& statistics = automatic_session.value()->statistics();
        check(static_cast<bool>(statistics.vulkan_linear_dispatches == 1));
        check(static_cast<bool>(statistics.vulkan_compute_submissions == 1));
        check(static_cast<bool>(statistics.vulkan_batch_uploads == 1));
        check(static_cast<bool>(statistics.vulkan_batch_downloads == 1));
        check(static_cast<bool>(statistics.vulkan_auxiliary_uploads == 0));
        check(static_cast<bool>(statistics.vulkan_auxiliary_upload_bytes == 0));
        check(static_cast<bool>(statistics.vulkan_staging_slot_resizes + statistics.vulkan_staging_slot_reuses == 2));
        check(static_cast<bool>(statistics.vulkan_staging_slot_acquisitions == 1));
    }
    else
        check(static_cast<bool>(automatic_session.value()->statistics().vulkan_linear_dispatches == 0));

    RuntimeOptions cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto cpu_model = runtime.load_model(package.path(), cpu_options);
    check(static_cast<bool>(cpu_model));
    check(static_cast<bool>(cpu_model.value()->vulkan_device_index() == automatic_vulkan_device_index));
    auto cpu_session = runtime.create_session(cpu_model.value());
    check(static_cast<bool>(cpu_session));
    auto cpu_prefill = cpu_session.value()->prefill(packed_prompt);
    check(static_cast<bool>(cpu_prefill));
    check(static_cast<bool>(cpu_prefill.value().logits.values.size() == automatic_prefill.value().logits.values.size()));
    for (size_t index = 0; index < cpu_prefill.value().logits.values.size(); ++index)
    {
        check_near(automatic_prefill.value().logits.values[index], cpu_prefill.value().logits.values[index], 1e-4f);
    }

    RuntimeOptions hybrid_options;
    hybrid_options.hybrid_mode = HybridMode::HybridExperts;
    auto hybrid_model = runtime.load_model(package.path(), hybrid_options);
    if (has_flag(runtime.capabilities().flags, RuntimeCapabilityVulkanCpuMix))
    {
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
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_attention_qkv_rope_fusions == 1));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_attention_qkv_ring_fusions == 0));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_attention_decode_sdpa_fusions == 0));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_linear_dispatches == 3));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_compute_submissions == 2));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_batch_uploads == 2));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_batch_downloads == 2));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_auxiliary_uploads == 3));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_auxiliary_upload_bytes > 0));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_staging_slot_resizes + attention_session.value()->statistics().vulkan_staging_slot_reuses == 7));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_staging_slot_reuses > 0));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_staging_slot_acquisitions == 2));
        check(static_cast<bool>(attention_session.value()->statistics().kv_cache_logical_bytes > 0));
        check(static_cast<bool>(attention_session.value()->statistics().kv_cache_allocated_bytes >= attention_session.value()->statistics().kv_cache_logical_bytes));

        auto cpu_attention_model = runtime.load_model(attention_package.path(), cpu_options);
        check(static_cast<bool>(cpu_attention_model));
        auto cpu_attention_session = runtime.create_session(cpu_attention_model.value());
        check(static_cast<bool>(cpu_attention_session));
        auto cpu_attention_prefill = cpu_attention_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(cpu_attention_prefill));
        check(static_cast<bool>(cpu_attention_prefill.value().logits.values.size() == attention_prefill.value().logits.values.size()));
        for (size_t index = 0; index < cpu_attention_prefill.value().logits.values.size(); ++index)
        {
            check_near(attention_prefill.value().logits.values[index], cpu_attention_prefill.value().logits.values[index], 1e-4f);
        }
        {
            const ScopedEnvironmentOverride force_decode_sdpa("NCNN_MOE_VULKAN_DECODE_SDPA", "1");
            auto attention_decode = attention_session.value()->decode(1);
            auto cpu_attention_decode = cpu_attention_session.value()->decode(1);
            check(static_cast<bool>(attention_decode));
            check(static_cast<bool>(cpu_attention_decode));
            for (size_t index = 0; index < cpu_attention_decode.value().logits.values.size(); ++index)
            {
                check_near(attention_decode.value().logits.values[index], cpu_attention_decode.value().logits.values[index], 1e-4f);
            }
            for (uint32_t iteration = 0; iteration < 20; ++iteration)
            {
                auto wrapped_decode = attention_session.value()->decode(static_cast<int32_t>(iteration % 2));
                auto wrapped_cpu_decode = cpu_attention_session.value()->decode(static_cast<int32_t>(iteration % 2));
                check(static_cast<bool>(wrapped_decode));
                check(static_cast<bool>(wrapped_cpu_decode));
                for (size_t index = 0; index < wrapped_cpu_decode.value().logits.values.size(); ++index)
                {
                    check_near(wrapped_decode.value().logits.values[index], wrapped_cpu_decode.value().logits.values[index], 1e-4f);
                }
            }
            const SessionStatistics& wrapped_statistics = attention_session.value()->statistics();
            check(static_cast<bool>(wrapped_statistics.vulkan_kv_ring_appends == wrapped_statistics.vulkan_attention_blocks));
            check(static_cast<bool>(wrapped_statistics.vulkan_attention_qkv_rope_fusions > 0));
            check(static_cast<bool>(wrapped_statistics.vulkan_attention_qkv_rope_fusions <= wrapped_statistics.vulkan_attention_blocks));
            check(static_cast<bool>(wrapped_statistics.vulkan_attention_qkv_ring_fusions == wrapped_statistics.vulkan_attention_blocks - 1));
            check(static_cast<bool>(wrapped_statistics.vulkan_attention_decode_sdpa_fusions == wrapped_statistics.vulkan_attention_blocks - 1));
            check(static_cast<bool>(wrapped_statistics.vulkan_kv_ring_resizes == 1));
            check(static_cast<bool>(wrapped_statistics.vulkan_kv_ring_wrapped_views > 0));
        }

        auto ncnn_sdpa_ring_session = runtime.create_session(attention_model.value());
        check(static_cast<bool>(ncnn_sdpa_ring_session));
        auto ncnn_sdpa_ring_prefill = ncnn_sdpa_ring_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(ncnn_sdpa_ring_prefill));
        auto ncnn_sdpa_ring_cpu_session = runtime.create_session(cpu_attention_model.value());
        check(static_cast<bool>(ncnn_sdpa_ring_cpu_session));
        auto ncnn_sdpa_ring_cpu_prefill = ncnn_sdpa_ring_cpu_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(ncnn_sdpa_ring_cpu_prefill));
        {
            const ScopedEnvironmentOverride disable_decode_sdpa("NCNN_MOE_VULKAN_DECODE_SDPA", "0");
            const ScopedEnvironmentOverride force_qkv_ring("NCNN_MOE_VULKAN_QKV_RING", "1");
            auto ncnn_sdpa_ring_decode = ncnn_sdpa_ring_session.value()->decode(1);
            auto ncnn_sdpa_ring_cpu_decode = ncnn_sdpa_ring_cpu_session.value()->decode(1);
            check(static_cast<bool>(ncnn_sdpa_ring_decode));
            check(static_cast<bool>(ncnn_sdpa_ring_cpu_decode));
            for (size_t index = 0; index < ncnn_sdpa_ring_cpu_decode.value().logits.values.size(); ++index)
            {
                check_near(ncnn_sdpa_ring_decode.value().logits.values[index], ncnn_sdpa_ring_cpu_decode.value().logits.values[index], 1e-4f);
            }
            check(static_cast<bool>(ncnn_sdpa_ring_session.value()->statistics().vulkan_attention_qkv_ring_fusions == 1));
            check(static_cast<bool>(ncnn_sdpa_ring_session.value()->statistics().vulkan_attention_decode_sdpa_fusions == 0));
        }

        AttentionPackage full_attention_package(false, 0);
        auto full_attention_model = runtime.load_model(full_attention_package.path(), hybrid_options);
        check(static_cast<bool>(full_attention_model));
        SessionOptions chunked_attention_options;
        chunked_attention_options.prefill_chunk_size = 2;
        auto chunked_attention_session = runtime.create_session(full_attention_model.value(), chunked_attention_options);
        check(static_cast<bool>(chunked_attention_session));
        auto chunked_attention_prefill = chunked_attention_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(chunked_attention_prefill));
        check(static_cast<bool>(chunked_attention_session.value()->statistics().kv_cache_allocated_bytes > chunked_attention_session.value()->statistics().kv_cache_logical_bytes));

        auto full_cpu_model = runtime.load_model(full_attention_package.path(), cpu_options);
        check(static_cast<bool>(full_cpu_model));
        auto full_cpu_session = runtime.create_session(full_cpu_model.value());
        check(static_cast<bool>(full_cpu_session));
        auto full_cpu_prefill = full_cpu_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(full_cpu_prefill));
        for (size_t index = 0; index < full_cpu_prefill.value().logits.values.size(); ++index)
        {
            check_near(chunked_attention_prefill.value().logits.values[index], full_cpu_prefill.value().logits.values[index], 1e-4f);
        }
        {
            const ScopedEnvironmentOverride force_decode_sdpa("NCNN_MOE_VULKAN_DECODE_SDPA", "1");
            auto chunked_attention_decode = chunked_attention_session.value()->decode(1);
            auto full_cpu_decode = full_cpu_session.value()->decode(1);
            check(static_cast<bool>(chunked_attention_decode));
            check(static_cast<bool>(full_cpu_decode));
            for (size_t index = 0; index < full_cpu_decode.value().logits.values.size(); ++index)
            {
                check_near(chunked_attention_decode.value().logits.values[index], full_cpu_decode.value().logits.values[index], 1e-4f);
            }
            check(static_cast<bool>(chunked_attention_session.value()->statistics().vulkan_attention_qkv_ring_fusions == 1));
            check(static_cast<bool>(chunked_attention_session.value()->statistics().vulkan_attention_decode_sdpa_fusions == 1));
        }

        auto unfused_ring_session = runtime.create_session(full_attention_model.value());
        check(static_cast<bool>(unfused_ring_session));
        auto unfused_ring_prefill = unfused_ring_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(unfused_ring_prefill));
        {
            const ScopedEnvironmentOverride force_decode_sdpa("NCNN_MOE_VULKAN_DECODE_SDPA", "1");
            const ScopedEnvironmentOverride disable_qkv_ring("NCNN_MOE_VULKAN_QKV_RING", "0");
            auto unfused_ring_decode = unfused_ring_session.value()->decode(1);
            check(static_cast<bool>(unfused_ring_decode));
            check(static_cast<bool>(unfused_ring_session.value()->statistics().vulkan_attention_qkv_ring_fusions == 0));
            check(static_cast<bool>(unfused_ring_session.value()->statistics().vulkan_attention_decode_sdpa_fusions == 1));
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
    else
    {
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
    check(static_cast<bool>(!has_flag(runtime.flags, RuntimeOptionRouterPrediction)));
    check(static_cast<bool>(!has_flag(runtime.flags, RuntimeOptionForwardAwareCache)));
    check(static_cast<bool>(!has_flag(runtime.flags, RuntimeOptionRankAdaptivePrefetch)));
    check(static_cast<bool>(!has_flag(runtime.flags, RuntimeOptionCrossExpertReadCoalescing)));
    check(static_cast<bool>(!has_flag(runtime.flags, RuntimeOptionAsyncRouterPrediction)));

    RuntimeCapabilities capabilities;
    check(static_cast<bool>(has_flag(capabilities.flags, RuntimeCapabilityCpuExecution)));
    check(static_cast<bool>(has_flag(capabilities.flags, RuntimeCapabilityMxfp4CpuKernel)));
    check(static_cast<bool>(has_flag(capabilities.flags, RuntimeCapabilityCrossSessionScheduling)));

    SchedulerOptions scheduler;
    check(static_cast<bool>(!has_flag(scheduler.flags, SchedulerOptionPinWorkers)));
    check(static_cast<bool>(!has_flag(scheduler.flags, SchedulerOptionDisableStagedBatching)));
    check(static_cast<bool>(!has_flag(scheduler.flags, SchedulerOptionForceStagedBatching)));

    ExpertDispatchOptions dispatch;
    check(static_cast<bool>(has_flag(dispatch.flags, ExpertDispatchNormalizeTopKWeights)));

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

void test_cpu_task_worker()
{
    CpuTaskWorker worker(2);
    std::mutex mutex;
    std::condition_variable started_condition;
    std::condition_variable release_condition;
    bool release = false;
    uint32_t started = 0;
    auto blocking_task = [&] {
        std::unique_lock<std::mutex> lock(mutex);
        ++started;
        started_condition.notify_all();
        release_condition.wait(lock, [&] {
            return release;
        });
    };

    check(worker.try_submit(blocking_task));
    check(worker.try_submit(blocking_task));
    check(!worker.try_submit([] {}));
    {
        std::unique_lock<std::mutex> lock(mutex);
        started_condition.wait(lock, [&] {
            return started != 0;
        });
        release = true;
    }
    release_condition.notify_all();
    worker.wait_idle();
    check(worker.completed_tasks() == 2);

    std::atomic<uint32_t> drained = 0;
    {
        CpuTaskWorker draining_worker(1);
        check(draining_worker.try_submit([&] {
            drained.fetch_add(1, std::memory_order_relaxed);
        }));
    }
    check(drained.load(std::memory_order_relaxed) == 1);
}

void test_float_scaled_add()
{
    std::array<float, 17> output = {};
    std::array<float, 17> input = {};
    std::array<float, 17> expected = {};
    constexpr float scale = -0.375f;
    for (size_t index = 0; index < output.size(); ++index)
    {
        output[index] = static_cast<float>(index) * 0.125f - 0.75f;
        input[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.25f;
        expected[index] = output[index] + scale * input[index];
    }
    float_scaled_add(
        output.data(),
        input.data(),
        scale,
        static_cast<uint32_t>(output.size()));
    for (size_t index = 0; index < output.size(); ++index)
        check_near(output[index], expected[index], 1e-6f);
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        ncnn::moe::test_flag_defaults();
        ncnn::moe::test_cpu_task_worker();
        ncnn::moe::test_float_scaled_add();
        ncnn::moe::test_ncnn_linear_operator();
        ncnn::moe::test_ncnn_vulkan_bfloat16_operator();
        ncnn::moe::test_ncnn_vulkan_float8_operator();
        ncnn::moe::test_mxfp4_cpu_kernel_and_fused_gate_up();
        ncnn::moe::test_sharded_expert_victim_cache();
        ncnn::moe::test_mapped_file_range_and_shared_buffer();
        ncnn::moe::test_safetensors_dense_mmap();
        ncnn::moe::test_safetensors_packed_mxfp4_expert();
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
        ncnn::moe::test_deepseek_router_and_hyper_connection_kernels();
        ncnn::moe::test_deepseek_v4_descriptors();
        ncnn::moe::test_qwen3_5_moe_descriptors();
        ncnn::moe::test_gated_delta_net_continuation();
        ncnn::moe::test_automatic_expert_memory_planning();
        ncnn::moe::test_sampling_and_streaming_generation();
        ncnn::moe::test_model_adapter_scopes();
        ncnn::moe::test_loader_reports_adapter_and_weight_errors();
        ncnn::moe::test_backend_capabilities_and_hybrid_execution();
        ncnn::moe::test_phase_zero_rejects_unimplemented_output_mode();
        std::cout << "all ncnn_moe tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
