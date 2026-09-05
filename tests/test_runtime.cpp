#include "ncnn/moe/runtime.h"

#include "graph/graph.h"
#include "graph/compiler.h"
#include "kernels/attention.h"
#include "kernels/activation.h"
#include "kernels/bfloat16.h"
#include "kernels/fastmath.h"
#include "kernels/gateddeltanet.h"
#include "kernels/gatedresidual.h"
#include "kernels/mxfp4.h"
#include "kernels/ple.h"
#include "kernels/qnk.h"
#include "kernels/ops.h"
#include "kernels/statecache.h"
#include "kernels/float8.h"
#include "kernels/vector.h"
#include "kernels/hyperconnection.h"
#include "engine/executor.h"
#include "engine/cpu.h"
#include "engine/expertbackend.h"
#include "graph/compiledmodel.h"
#include "engine/sessionstate.h"
#include "storage/expertcache.h"
#include "modeladapter_fixture.h"
#include "storage/mappedfile.h"
#include "graph/memoryplan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/modelpipeline.h"
#include "backends/ncnn/expertbackend_vulkan.h"
#include "graph/router.h"
#include "models/modeladapter_builtin.h"
#include "models/modeladapter.h"
#include "models/modeladapter_deepseekv4.h"
#include "models/modeladapter_qwen3_5.h"
#include "models/modeladapter_qwen4exp.h"
#include "models/safetensors.h"

#if defined(_MSC_VER) && defined(_M_X64)
#include "kernels/vector_msvc.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
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

static void check_near(
    float actual,
    float expected,
    float tolerance,
    const std::source_location location = std::source_location::current())
{
    if (std::abs(actual - expected) > tolerance)
    {
        throw std::runtime_error(
            "near check failed at " + std::string(location.file_name()) + ":"
            + std::to_string(location.line()) + ": actual="
            + std::to_string(actual) + ", expected=" + std::to_string(expected));
    }
}

static float bfloat16_storage_tolerance(float value) noexcept
{
    // Native conversion may round, while ncnn's packed fallback truncates.
    // One storage ULP is the strict bound shared by both paths.
    if (value == 0.0f)
        return std::ldexp(1.0f, -133);
    return std::ldexp(
        1.0f,
        std::max(std::ilogb(std::abs(value)) - 7, -133));
}

inline uint64_t g_test_optimization_flags = OptimizationDefaultFlags;

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

class Bfloat16StagedBatchPackage
{
public:
    Bfloat16StagedBatchPackage()
        : path_(create_unique_test_directory("ncnn_moe_bfloat16_staged_test_"))
    {
        std::ofstream manifest(path_ / "config.json", std::ios::binary);
        manifest << R"({
  "model_type": "test_moe",
  "vocabulary_size": 4,
  "hidden_size": 512,
  "intermediate_size": 512,
  "layer_count": 1,
  "expert_count": 1,
  "experts_per_token": 1,
  "expert_activation": "relu",
  "expert_layout": "up_down",
  "expert_weight_dtype": "bfloat16",
  "normalize_topk_weights": true,
  "norm_epsilon": 0.00001,
  "weights_file": "model.test.bin"
})";
        manifest.close();

        std::ofstream weights(path_ / "model.test.bin", std::ios::binary);
        std::vector<float> token_embedding(4 * 512);
        for (size_t index = 0; index < token_embedding.size(); ++index)
        {
            token_embedding[index] = static_cast<float>(static_cast<int>((index * 7 + 3) % 17) - 8)
                                     * 0.0625f;
        }
        write_floats(weights, token_embedding);
        write_floats(weights, std::vector<float>(512, 1.0f));
        write_floats(weights, std::vector<float>(512, 0.0f));

        std::vector<uint16_t> identity(512 * 512, float_to_bfloat16(0.0f));
        for (uint32_t row = 0; row < 512; ++row)
            identity[static_cast<size_t>(row) * 512 + row] = float_to_bfloat16(1.0f);
        write_bfloat16(weights, identity);
        write_bfloat16(weights, identity);

        write_floats(weights, std::vector<float>(512, 1.0f));
        std::vector<float> lm_head(4 * 512);
        for (size_t index = 0; index < lm_head.size(); ++index)
        {
            lm_head[index] = static_cast<float>(static_cast<int>((index * 5 + 1) % 13) - 6)
                             * 0.03125f;
        }
        write_floats(weights, lm_head);
    }

    ~Bfloat16StagedBatchPackage()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    static void write_floats(
        std::ofstream& stream,
        const std::vector<float>& values)
    {
        stream.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    static void write_bfloat16(
        std::ofstream& stream,
        const std::vector<uint16_t>& values)
    {
        stream.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(uint16_t)));
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
    check(static_cast<bool>(has_flag(runtime.info().flags, RuntimeVulkanAttention) == has_flag(runtime.info().flags, RuntimeVulkanCpu)));
    check(static_cast<bool>(runtime.info().cpu_linear_num_threads >= 1));
    check(static_cast<bool>(runtime.info().float8_linear_row_group_size == 1 || runtime.info().float8_linear_row_group_size == 2
                            || runtime.info().float8_linear_row_group_size == 4
                            || runtime.info().float8_linear_row_group_size == 8));
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    check(static_cast<bool>(model.value()->descriptor().model_type == "test_moe"));
    check(static_cast<bool>(model.value()->descriptor().layers.size() == 1));

    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));

    const std::vector<int32_t> prompt = {0, 1, 2};
    auto prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill));
    check(static_cast<bool>(prefill.value().processed_tokens == 3));
    check(static_cast<bool>(prefill.value().logits.size() == 4));

    const float normalized_equal = (1.0f + 1.0f / std::sqrt(1.0f + 1e-5f));
    const float final_value = normalized_equal / std::sqrt(normalized_equal * normalized_equal + 1e-5f);
    check_near(prefill.value().logits[0], final_value, 1e-5f);
    check_near(prefill.value().logits[1], final_value, 1e-5f);
    check_near(prefill.value().logits[2], 2.0f * final_value, 1e-5f);
    check_near(prefill.value().logits[3], -final_value, 1e-5f);

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
    check_near(decode.value().logits[0], 0.0f, 1e-5f);
    check_near(decode.value().logits[1], final_expert_one, 2e-5f);
    check_near(decode.value().logits[2], final_expert_one, 2e-5f);
    check_near(decode.value().logits[3], 0.0f, 1e-5f);

    check(static_cast<bool>(session.value()->reset()));
    check(static_cast<bool>(session.value()->sequence_length() == 0));
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 0));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 0));
    check(static_cast<bool>(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({0, 0})));
}

void test_ncnn_linear_operator()
{
#if NCNN_MOE_WITH_NCNN
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
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

    const auto linear = NcnnLinearOperator::create(
        matrix,
        &bias,
        NcnnLinearDevice::Cpu,
        automatic_vulkan_device_index,
        context_instance,
        g_test_optimization_flags);
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

    {
        const uint64_t direct_bfloat16_flags = g_test_optimization_flags
                                               & ~OptimizationNcnnCpuBfloat16Linear;
        check(std::string(
                  NcnnLinearOperator::cpu_small_bfloat16_linear_policy(
                      direct_bfloat16_flags))
              == "moe-direct-bfloat16");
        check(!NcnnLinearOperator::create(
            bfloat_matrix,
            &bfloat_bias,
            NcnnLinearDevice::Cpu,
            automatic_vulkan_device_index,
            context_instance,
            direct_bfloat16_flags));
    }
    {
        const uint64_t ncnn_bfloat16_flags = g_test_optimization_flags
                                             | OptimizationNcnnCpuBfloat16Linear;
        const bool ncnn_bfloat16_enabled = std::string(
                                               NcnnLinearOperator::cpu_small_bfloat16_linear_policy(
                                                   ncnn_bfloat16_flags))
                                           == "ncnn-fp32-expanded";
        const auto bfloat_linear = NcnnLinearOperator::create(
            bfloat_matrix,
            &bfloat_bias,
            NcnnLinearDevice::Cpu,
            automatic_vulkan_device_index,
            context_instance,
            ncnn_bfloat16_flags);
        check(static_cast<bool>(bfloat_linear) == ncnn_bfloat16_enabled);
        if (bfloat_linear)
        {
            check(static_cast<bool>(bfloat_linear->forward(input, output)));
            for (size_t row_index = 0; row_index < input.rows(); ++row_index)
            {
                for (uint32_t column = 0; column < output.columns(); ++column)
                {
                    float expected = bfloat16_to_float(
                        bfloat_bias.bfloat16_data[column]);
                    for (uint32_t input_column = 0;
                         input_column < input.columns();
                         ++input_column)
                    {
                        expected += bfloat16_to_float(
                                        bfloat_matrix.bfloat16_data[column * input.columns()
                                                                    + input_column])
                                    * input.row(row_index)[input_column];
                    }
                    check_near(output.row(row_index)[column], expected, 1e-5f);
                }
            }
        }
    }

    if (get_gpu_count() > 0)
    {
        const auto vulkan_linear = NcnnLinearOperator::create(
            matrix,
            &bias,
            NcnnLinearDevice::Vulkan,
            automatic_vulkan_device_index,
            context_instance,
            g_test_optimization_flags);
        check(static_cast<bool>(vulkan_linear));
        const NcnnVulkanRuntimeCounters initial_counters = get_vulkan_execution_snapshot(context_instance).counters;
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
        const NcnnVulkanRuntimeCounters final_counters = get_vulkan_execution_snapshot(context_instance).counters;
        check(static_cast<bool>(final_counters.compute_submissions - initial_counters.compute_submissions == 4));
        check(static_cast<bool>(final_counters.batch_uploads - initial_counters.batch_uploads == 4));
        check(static_cast<bool>(final_counters.batch_downloads - initial_counters.batch_downloads == 4));
        check(static_cast<bool>(final_counters.auxiliary_uploads - initial_counters.auxiliary_uploads == 0));
        check(static_cast<bool>(final_counters.staging_slot_resizes - initial_counters.staging_slot_resizes + final_counters.staging_slot_reuses - initial_counters.staging_slot_reuses == 8));
        check(static_cast<bool>(final_counters.staging_slot_reuses - initial_counters.staging_slot_reuses >= 4));
        check(static_cast<bool>(final_counters.staging_slot_acquisitions - initial_counters.staging_slot_acquisitions == 4));
        const uint64_t command_dispatches = final_counters.command_dispatches
                                            - initial_counters.command_dispatches;
        const uint64_t command_pipeline_binds = final_counters.command_pipeline_binds
                                                - initial_counters.command_pipeline_binds;
        const uint64_t command_descriptor_bindings = final_counters.command_descriptor_bindings
                                                     - initial_counters.command_descriptor_bindings;
        const uint64_t command_redundant_pipeline_binds = final_counters.command_redundant_pipeline_binds
                                                          - initial_counters.command_redundant_pipeline_binds;
        const uint64_t command_push_constant_updates = final_counters.command_push_constant_updates
                                                       - initial_counters.command_push_constant_updates;
        const uint64_t command_resource_barrier_calls = final_counters.command_resource_barrier_calls
                                                        - initial_counters.command_resource_barrier_calls;
        const uint64_t command_buffer_resource_barriers = final_counters.command_buffer_resource_barriers
                                                          - initial_counters.command_buffer_resource_barriers;
        const uint64_t command_image_resource_barriers = final_counters.command_image_resource_barriers
                                                         - initial_counters.command_image_resource_barriers;
        check(static_cast<bool>(command_dispatches > 0));
        check(static_cast<bool>(command_pipeline_binds == command_dispatches));
        check(static_cast<bool>(
            command_redundant_pipeline_binds <= command_dispatches));
        check(static_cast<bool>(command_descriptor_bindings > 0));
        check(static_cast<bool>(
            command_descriptor_bindings <= command_dispatches));
        check(static_cast<bool>(command_push_constant_updates > 0));
        check(static_cast<bool>(
            command_push_constant_updates <= command_dispatches));
        check(static_cast<bool>(
            command_resource_barrier_calls
            == command_buffer_resource_barriers
                   + command_image_resource_barriers));

        TensorData chain_matrix;
        chain_matrix.dtype = DType::Float32;
        chain_matrix.shape = {5, 4};
        chain_matrix.float32_data.resize(chain_matrix.element_count());
        for (size_t index = 0; index < chain_matrix.float32_data.size(); ++index)
        {
            chain_matrix.float32_data[index] = static_cast<float>(static_cast<int>((index * 7) % 23) - 11)
                                               * 0.03125f;
        }
        TensorData parallel_matrix;
        parallel_matrix.dtype = DType::Float32;
        parallel_matrix.shape = {6, 3};
        parallel_matrix.float32_data.resize(parallel_matrix.element_count());
        for (size_t index = 0;
             index < parallel_matrix.float32_data.size();
             ++index)
        {
            parallel_matrix.float32_data[index] = static_cast<float>(static_cast<int>((index * 11) % 29) - 14)
                                                  * 0.0234375f;
        }
        const auto chain_operator = NcnnLinearOperator::create(
            chain_matrix,
            nullptr,
            NcnnLinearDevice::Vulkan,
            automatic_vulkan_device_index,
            context_instance,
            g_test_optimization_flags);
        const auto parallel_operator = NcnnLinearOperator::create(
            parallel_matrix,
            nullptr,
            NcnnLinearDevice::Vulkan,
            automatic_vulkan_device_index,
            context_instance,
            g_test_optimization_flags);
        check(static_cast<bool>(chain_operator));
        check(static_cast<bool>(parallel_operator));
        const CpuBatch expected_chain = linear_batch(
            chain_matrix,
            expected_output,
            g_test_optimization_flags);
        const CpuBatch expected_parallel = linear_batch(
            parallel_matrix,
            input,
            g_test_optimization_flags);
        const NcnnVulkanRuntimeCounters graph_before = get_vulkan_execution_snapshot(context_instance).counters;
        auto graph = NcnnVulkanCommandGraph::create(*vulkan_linear);
        NcnnVulkanDeviceTensor graph_input;
        NcnnVulkanDeviceTensor graph_first;
        NcnnVulkanDeviceTensor graph_chain;
        NcnnVulkanDeviceTensor graph_parallel;
        CpuBatch graph_chain_output;
        CpuBatch graph_parallel_output;
        check(static_cast<bool>(
            graph
            && graph->upload(input, graph_input)
            && graph->linear(*vulkan_linear, graph_input, graph_first)
            && graph->linear(*chain_operator, graph_first, graph_chain)
            && graph->linear(
                *parallel_operator,
                graph_input,
                graph_parallel)
            && graph->download(graph_chain, graph_chain_output)
            && graph->download(graph_parallel, graph_parallel_output)
            && graph->submit()
            && graph->wait()));
        for (size_t row = 0; row < input.rows(); ++row)
        {
            for (uint32_t column = 0;
                 column < expected_chain.columns();
                 ++column)
            {
                check_near(
                    graph_chain_output.row(row)[column],
                    expected_chain.row(row)[column],
                    1e-4f);
            }
            for (uint32_t column = 0;
                 column < expected_parallel.columns();
                 ++column)
            {
                check_near(
                    graph_parallel_output.row(row)[column],
                    expected_parallel.row(row)[column],
                    1e-4f);
            }
        }
        const NcnnVulkanRuntimeCounters graph_after = get_vulkan_execution_snapshot(context_instance).counters;
        check(static_cast<bool>(
            graph_after.compute_submissions
                - graph_before.compute_submissions
            == 1));
        check(static_cast<bool>(
            graph_after.batch_uploads - graph_before.batch_uploads == 1));
        check(static_cast<bool>(
            graph_after.batch_downloads - graph_before.batch_downloads == 2));
        check(static_cast<bool>(
            graph_after.command_graph_submissions
                - graph_before.command_graph_submissions
            == 1));
        check(static_cast<bool>(
            graph_after.command_graph_operations
                - graph_before.command_graph_operations
            == 4));

        CpuBatch typed_input(input.rows(), input.columns(), DType::BFloat16);
        for (size_t row = 0; row < input.rows(); ++row)
        {
            const std::span<std::byte> destination = typed_input.mutable_row_bytes(row);
            for (uint32_t column = 0; column < input.columns(); ++column)
            {
                const uint16_t value = float_to_bfloat16(
                    input.row(row)[column]);
                std::memcpy(
                    destination.data() + column * sizeof(uint16_t),
                    &value,
                    sizeof(value));
            }
        }
        CpuBatch typed_output(0, 0, DType::BFloat16);
        auto typed_graph = NcnnVulkanCommandGraph::create(*vulkan_linear);
        NcnnVulkanDeviceTensor typed_graph_input;
        NcnnVulkanDeviceTensor typed_graph_output;
        check(static_cast<bool>(typed_graph));
        check(typed_graph->upload(typed_input, typed_graph_input));
        check(typed_graph->linear(
            *vulkan_linear,
            typed_graph_input,
            typed_graph_output));
        check(typed_graph->download(typed_graph_output, typed_output));
        check(typed_graph->submit());
        check(typed_graph->wait());
        check(static_cast<bool>(typed_output.dtype() == DType::BFloat16));
        for (size_t row = 0; row < typed_input.rows(); ++row)
        {
            const std::span<const std::byte> source = typed_input.row_bytes(row);
            for (uint32_t column = 0;
                 column < typed_input.columns();
                 ++column)
            {
                float expected = bias.float32_data[column];
                for (uint32_t input_column = 0;
                     input_column < typed_input.columns();
                     ++input_column)
                {
                    uint16_t input_value = 0;
                    std::memcpy(
                        &input_value,
                        source.data()
                            + input_column * sizeof(uint16_t),
                        sizeof(input_value));
                    expected += matrix.float32_data[column * typed_input.columns()
                                                    + input_column]
                                * bfloat16_to_float(input_value);
                }
                const std::span<const std::byte> output_bytes = typed_output.row_bytes(row);
                uint16_t output_value = 0;
                std::memcpy(
                    &output_value,
                    output_bytes.data() + column * sizeof(uint16_t),
                    sizeof(output_value));
                check_near(
                    bfloat16_to_float(output_value),
                    expected,
                    bfloat16_storage_tolerance(expected));
            }
        }
    }
#endif
}

void test_dense_mxn_tiles()
{
    TensorData float_matrix;
    float_matrix.dtype = DType::Float32;
    float_matrix.shape = {8, 32};
    float_matrix.float32_data.resize(float_matrix.element_count());
    for (size_t index = 0; index < float_matrix.float32_data.size(); ++index)
    {
        float_matrix.float32_data[index] = static_cast<float>(static_cast<int>((index * 13 + 3) % 41) - 20)
                                           * 0.03125f;
    }
    CpuBatch input(8, 32);
    for (size_t index = 0; index < input.rows() * input.columns(); ++index)
    {
        input.row(index / input.columns())[index % input.columns()] = static_cast<float>(static_cast<int>((index * 7 + 5) % 37) - 18)
                                                                      * 0.015625f;
    }
    const CpuBatch float_output = linear_batch(
        float_matrix,
        input,
        g_test_optimization_flags);
    for (size_t token = 0; token < input.rows(); ++token)
    {
        for (uint32_t output_column = 0;
             output_column < float_matrix.shape[0];
             ++output_column)
        {
            float expected = 0.0f;
            for (uint32_t input_column = 0;
                 input_column < input.columns();
                 ++input_column)
            {
                expected += float_matrix.float32_data[static_cast<size_t>(output_column) * input.columns()
                                                      + input_column]
                            * input.row(token)[input_column];
            }
            check_near(
                float_output.row(token)[output_column],
                expected,
                1e-4f);
        }
    }

    TensorData bfloat_matrix;
    bfloat_matrix.dtype = DType::BFloat16;
    bfloat_matrix.shape = float_matrix.shape;
    bfloat_matrix.bfloat16_data.reserve(float_matrix.float32_data.size());
    for (float value : float_matrix.float32_data)
        bfloat_matrix.bfloat16_data.push_back(float_to_bfloat16(value));
    const CpuBatch bfloat_output = linear_batch(
        bfloat_matrix,
        input,
        g_test_optimization_flags);
    for (size_t token = 0; token < input.rows(); ++token)
    {
        for (uint32_t output_column = 0;
             output_column < bfloat_matrix.shape[0];
             ++output_column)
        {
            float expected = 0.0f;
            for (uint32_t input_column = 0;
                 input_column < input.columns();
                 ++input_column)
            {
                expected += bfloat16_to_float(
                                bfloat_matrix.bfloat16_data[static_cast<size_t>(output_column) * input.columns()
                                                            + input_column])
                            * input.row(token)[input_column];
            }
            check_near(
                bfloat_output.row(token)[output_column],
                expected,
                1e-4f);
        }
    }
}

void test_released_dense_host_storage_guard()
{
    CpuBatch input(1, 2);
    input.row(0)[0] = 1.0f;
    input.row(0)[1] = -2.0f;

    TensorData released_matrix;
    released_matrix.dtype = DType::Float32;
    released_matrix.shape = {2, 2};
    CpuBatch output;
    bool matrix_failure_reported = false;
    try
    {
        linear_batch_into(
            released_matrix,
            input,
            output,
            g_test_optimization_flags);
    }
    catch (const std::runtime_error& error)
    {
        matrix_failure_reported = std::string(error.what()).find("host storage was released")
                                  != std::string::npos;
    }
    check(matrix_failure_reported);

    TensorData matrix;
    matrix.dtype = DType::Float32;
    matrix.shape = {2, 2};
    matrix.float32_data = {1.0f, 0.0f, 0.0f, 1.0f};
    TensorData released_bias;
    released_bias.dtype = DType::Float32;
    released_bias.shape = {2};
    bool bias_failure_reported = false;
    try
    {
        linear_batch_into(
            matrix,
            released_bias,
            input,
            output,
            g_test_optimization_flags);
    }
    catch (const std::runtime_error& error)
    {
        bias_failure_reported = std::string(error.what()).find("host storage was released")
                                != std::string::npos;
    }
    check(bias_failure_reported);

    TensorData released_norm;
    released_norm.dtype = DType::BFloat16;
    released_norm.shape = {2};
    bool norm_failure_reported = false;
    try
    {
        rms_norm_batch_into(
            input,
            released_norm,
            1e-5f,
            output,
            0.0f,
            g_test_optimization_flags);
    }
    catch (const std::runtime_error& error)
    {
        norm_failure_reported = std::string(error.what()).find("host storage was released")
                                != std::string::npos;
    }
    check(norm_failure_reported);
}

void test_ncnn_vulkan_float8_operator()
{
#if NCNN_MOE_WITH_NCNN
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
    if (get_gpu_count() == 0)
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
    matrix.mapped_size = element_count;
    matrix.quantization_scales = {0.5f, 2.0f};

    CpuBatch input(2, 128);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
        {
            input.row(row)[column] = static_cast<float>(static_cast<int>((row * input.columns() + column) % 37) - 18) * 0.015625f;
        }
    }
    const CpuBatch cpu_output = linear_batch(
        matrix,
        input,
        g_test_optimization_flags);
    const auto vulkan = NcnnVulkanFloat8Operator::create(
        matrix,
        nullptr,
        1,
        automatic_vulkan_device_index,
        context_instance,
        g_test_optimization_flags);
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
    second_matrix.mapped_size = second_element_count;
    second_matrix.quantization_scales = {0.5f, 1.5f};
    const CpuBatch cpu_chain = linear_batch(
        second_matrix,
        cpu_output,
        g_test_optimization_flags);
    const auto second_vulkan = NcnnVulkanFloat8Operator::create(
        second_matrix,
        nullptr,
        1,
        automatic_vulkan_device_index,
        context_instance,
        g_test_optimization_flags);
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
    const CpuBatch normalized_cpu_output = rms_norm_batch(
        cpu_output,
        norm_weight,
        norm_epsilon,
        0.0f,
        g_test_optimization_flags);
    const CpuBatch cpu_norm_chain = linear_batch(
        second_matrix,
        normalized_cpu_output,
        g_test_optimization_flags);
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
    const NcnnVulkanRuntimeCounters parallel_counters_before = get_vulkan_execution_snapshot(context_instance).counters;
    check(static_cast<bool>(vulkan->forward_rms_norm_chain_parallel(
        input,
        *second_vulkan,
        *vulkan,
        vulkan_parallel_chain,
        vulkan_parallel_output)));
    const NcnnVulkanRuntimeCounters parallel_counters_after = get_vulkan_execution_snapshot(context_instance).counters;
    const uint64_t parallel_dispatches = parallel_counters_after.command_dispatches
                                         - parallel_counters_before.command_dispatches;
    const uint64_t parallel_pipeline_binds = parallel_counters_after.command_pipeline_binds
                                             - parallel_counters_before.command_pipeline_binds;
    const uint64_t parallel_redundant_pipeline_binds = parallel_counters_after.command_redundant_pipeline_binds
                                                       - parallel_counters_before.command_redundant_pipeline_binds;
    const bool bind_elision_enabled = has_flag(
        g_test_optimization_flags,
        OptimizationVulkanPipelineBindElision);
    check(static_cast<bool>(parallel_redundant_pipeline_binds == 1));
    check(static_cast<bool>(
        parallel_pipeline_binds
            + (bind_elision_enabled
                   ? parallel_redundant_pipeline_binds
                   : 0)
        == parallel_dispatches));
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
    const CpuBatch cpu_swiglu_chain = linear_batch(
        second_matrix,
        cpu_activated,
        g_test_optimization_flags);
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
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
    if (get_gpu_count() == 0)
        return;
    TensorData first;
    first.dtype = DType::BFloat16;
    first.shape = {192, 128};
    first.bfloat16_data.resize(first.element_count());
    for (size_t index = 0; index < first.element_count(); ++index)
    {
        const float value = static_cast<float>(
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
        const float value = static_cast<float>(
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
        first_bias.bfloat16_data[index] = float_to_bfloat16(
            static_cast<float>(static_cast<int>(index % 13) - 6)
            * 0.0021f);
    }
    TensorData second_bias;
    second_bias.dtype = DType::Float32;
    second_bias.shape = {64};
    second_bias.float32_data.resize(64);
    for (uint32_t index = 0; index < 64; ++index)
    {
        second_bias.float32_data[index] = static_cast<float>(static_cast<int>(index % 9) - 4)
                                          * 0.0019f;
    }

    CpuBatch input(3, 128);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
        {
            input.row(row)[column] = static_cast<float>(
                                         static_cast<int>(
                                             (row * input.columns() + column * 7) % 89)
                                         - 44)
                                     * 0.0031f;
        }
    }
    const CpuBatch first_cpu = linear_batch(first, first_bias, input, g_test_optimization_flags);
    const auto first_vulkan = NcnnVulkanBfloat16Operator::create(
        first,
        &first_bias,
        automatic_vulkan_device_index,
        context_instance,
        g_test_optimization_flags);
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

    TensorData norm_weight;
    norm_weight.dtype = DType::Float32;
    norm_weight.shape = {128};
    norm_weight.float32_data.resize(128);
    for (uint32_t column = 0; column < 128; ++column)
    {
        norm_weight.float32_data[column] = 0.75f + static_cast<float>(column % 9) * 0.03125f;
    }
    constexpr float norm_epsilon = 1e-6f;
    const CpuBatch normalized_cpu = rms_norm_batch(
        input,
        norm_weight,
        norm_epsilon,
        0.0f,
        g_test_optimization_flags);
    const CpuBatch norm_chain_cpu = linear_batch(
        first,
        first_bias,
        normalized_cpu,
        g_test_optimization_flags);
    check(static_cast<bool>(
        first_vulkan->prepare_rms_norm(norm_weight, norm_epsilon)));
    CpuBatch norm_chain_vulkan;
    check(static_cast<bool>(first_vulkan->forward_rms_norm_chain(
        input,
        norm_chain_vulkan)));
    for (size_t row = 0; row < norm_chain_cpu.rows(); ++row)
    {
        for (uint32_t column = 0;
             column < norm_chain_cpu.columns();
             ++column)
        {
            check_near(
                norm_chain_vulkan.row(row)[column],
                norm_chain_cpu.row(row)[column],
                2e-3f);
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

    {
        const uint64_t cooperative_flags = g_test_optimization_flags | OptimizationVulkanBfloat16CoopMatrix;
        const auto cooperative_operator = NcnnVulkanBfloat16Operator::create(
            first,
            &first_bias,
            automatic_vulkan_device_index,
            context_instance,
            cooperative_flags);
        check(static_cast<bool>(cooperative_operator));
        CpuBatch cooperative_input(16, input.columns());
        for (size_t row = 0; row < cooperative_input.rows(); ++row)
        {
            std::copy_n(
                input.row(row % input.rows()),
                input.columns(),
                cooperative_input.row(row));
        }
        const CpuBatch cooperative_reference = linear_batch(
            first,
            first_bias,
            cooperative_input,
            cooperative_flags);
        CpuBatch cooperative_output;
        check(static_cast<bool>(cooperative_operator->forward(
            cooperative_input,
            cooperative_output)));
        for (size_t row = 0; row < cooperative_reference.rows(); ++row)
        {
            for (uint32_t column = 0;
                 column < cooperative_reference.columns();
                 ++column)
            {
                check_near(
                    cooperative_output.row(row)[column],
                    cooperative_reference.row(row)[column],
                    3e-3f);
            }
        }
    }

    {
        TensorData tail_matrix;
        tail_matrix.dtype = DType::BFloat16;
        tail_matrix.shape = {64, 192};
        tail_matrix.bfloat16_data.resize(tail_matrix.element_count());
        for (size_t index = 0;
             index < tail_matrix.bfloat16_data.size();
             ++index)
        {
            tail_matrix.bfloat16_data[index] = float_to_bfloat16(
                static_cast<float>(
                    static_cast<int>((index * 11) % 53) - 26)
                * 0.0017f);
        }
        CpuBatch tail_input(3, 192);
        for (size_t row = 0; row < tail_input.rows(); ++row)
        {
            for (uint32_t column = 0;
                 column < tail_input.columns();
                 ++column)
            {
                tail_input.row(row)[column] = static_cast<float>(
                                                  static_cast<int>((row * 17 + column * 5) % 71)
                                                  - 35)
                                              * 0.0023f;
            }
        }
        const CpuBatch tail_reference = linear_batch(
            tail_matrix,
            tail_input,
            g_test_optimization_flags);
        const auto tail_operator = NcnnVulkanBfloat16Operator::create(
            tail_matrix,
            nullptr,
            automatic_vulkan_device_index,
            context_instance,
            g_test_optimization_flags);
        check(static_cast<bool>(tail_operator));
        CpuBatch tail_output;
        check(static_cast<bool>(tail_operator->forward(
            tail_input,
            tail_output)));
        for (size_t row = 0; row < tail_reference.rows(); ++row)
        {
            for (uint32_t column = 0;
                 column < tail_reference.columns();
                 ++column)
            {
                check_near(
                    tail_output.row(row)[column],
                    tail_reference.row(row)[column],
                    3e-4f);
            }
        }
    }

    const auto fused = NcnnVulkanBfloat16Operator::create_fused(
        {&first, &second},
        {&first_bias, &second_bias},
        automatic_vulkan_device_index,
        context_instance,
        g_test_optimization_flags);
    check(static_cast<bool>(fused));
    CpuBatch fused_output;
    check(static_cast<bool>(
        fused->forward(input, fused_output)));
    const CpuBatch second_cpu = linear_batch(
        second,
        second_bias,
        input,
        g_test_optimization_flags);
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

    TensorData gate;
    gate.dtype = DType::BFloat16;
    gate.shape = {128, 128};
    gate.bfloat16_data.resize(gate.element_count());
    TensorData up;
    up.dtype = DType::BFloat16;
    up.shape = {128, 128};
    up.bfloat16_data.resize(up.element_count());
    TensorData down;
    down.dtype = DType::BFloat16;
    down.shape = {64, 128};
    down.bfloat16_data.resize(down.element_count());
    TensorData router_gate;
    router_gate.dtype = DType::BFloat16;
    router_gate.shape = {1, 128};
    router_gate.bfloat16_data.resize(router_gate.element_count());
    for (size_t index = 0; index < gate.element_count(); ++index)
    {
        gate.bfloat16_data[index] = float_to_bfloat16(
            static_cast<float>(static_cast<int>((index * 13) % 61) - 30) * 0.0023f);
        up.bfloat16_data[index] = float_to_bfloat16(
            static_cast<float>(static_cast<int>((index * 7) % 47) - 23) * 0.0027f);
    }
    for (size_t index = 0; index < down.element_count(); ++index)
    {
        down.bfloat16_data[index] = float_to_bfloat16(
            static_cast<float>(static_cast<int>((index * 19) % 73) - 36) * 0.0019f);
    }
    for (size_t index = 0; index < router_gate.element_count(); ++index)
    {
        router_gate.bfloat16_data[index] = float_to_bfloat16(
            static_cast<float>(static_cast<int>(index % 17) - 8) * 0.0031f);
    }
    const auto fused_swiglu = NcnnVulkanBfloat16Operator::create_fused(
        {&gate, &up, &router_gate},
        {nullptr, nullptr, nullptr},
        automatic_vulkan_device_index,
        context_instance,
        g_test_optimization_flags);
    const auto down_vulkan = NcnnVulkanBfloat16Operator::create(
        down,
        nullptr,
        automatic_vulkan_device_index,
        context_instance,
        g_test_optimization_flags);
    check(static_cast<bool>(fused_swiglu));
    check(static_cast<bool>(down_vulkan));
    const CpuBatch gate_cpu = linear_batch(
        gate,
        input,
        g_test_optimization_flags);
    const CpuBatch up_cpu = linear_batch(
        up,
        input,
        g_test_optimization_flags);
    const CpuBatch router_cpu = linear_batch(
        router_gate,
        input,
        g_test_optimization_flags);
    CpuBatch activated(input.rows(), 128);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < 128; ++column)
        {
            const float gate_value = gate_cpu.row(row)[column];
            activated.row(row)[column] = gate_value / (1.0f + std::exp(-gate_value))
                                         * up_cpu.row(row)[column];
        }
    }
    CpuBatch expected_swiglu = linear_batch(
        down,
        activated,
        g_test_optimization_flags);
    for (size_t row = 0; row < expected_swiglu.rows(); ++row)
    {
        const float router_scale = 1.0f / (1.0f + std::exp(-router_cpu.row(row)[0]));
        for (uint32_t column = 0; column < expected_swiglu.columns(); ++column)
            expected_swiglu.row(row)[column] *= router_scale;
    }
    CpuBatch actual_swiglu;
    const NcnnVulkanRuntimeCounters before_swiglu = get_vulkan_execution_snapshot(context_instance).counters;
    check(static_cast<bool>(fused_swiglu->forward_swiglu_chain(
        input,
        *down_vulkan,
        128,
        ExpertActivation::Silu,
        0.0f,
        true,
        actual_swiglu)));
    const NcnnVulkanRuntimeCounters after_swiglu = get_vulkan_execution_snapshot(context_instance).counters;
    check(static_cast<bool>(
        after_swiglu.shared_expert_swiglu_fusions
        == before_swiglu.shared_expert_swiglu_fusions + 1));
    for (size_t row = 0; row < expected_swiglu.rows(); ++row)
    {
        for (uint32_t column = 0; column < expected_swiglu.columns(); ++column)
            check_near(actual_swiglu.row(row)[column], expected_swiglu.row(row)[column], 3e-3f);
    }
#endif
}

void test_mxfp4_cpu_kernel_and_fused_gate_up()
{
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
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

    const CpuBatch projected = linear_batch(
        matrix,
        input,
        g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
    {
        for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
        {
            check_near(projected.row(input_row)[matrix_row], scalar_row(matrix_row, input_row), 1e-5f);
        }
    }
    const uint64_t q8_flags = (g_test_optimization_flags | OptimizationCpuMxfp4Q8)
                              & ~OptimizationCpuPackedWeights;
    check(!has_flag(q8_flags, OptimizationCpuPackedWeights));
    const CpuBatch unpacked_q8_projected = linear_batch(
        matrix,
        input,
        q8_flags);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
        for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
            check_near(unpacked_q8_projected.row(input_row)[matrix_row], scalar_row(matrix_row, input_row), 0.15f);
    const CpuBatch packed_q8_projected = linear_batch(
        matrix,
        input,
        q8_flags | OptimizationCpuPackedWeights);
    CompiledOperator packed_owner;
    CpuBatch owner_unpacked_q8_projected;
    linear_batch_into(
        matrix,
        input,
        owner_unpacked_q8_projected,
        q8_flags,
        &packed_owner,
        ExecutionBackend::Cpu);
    check(static_cast<bool>(!packed_owner.mxfp4_q8_packed));
    const CpuBatch owner_packed_q8_projected = linear_batch(
        matrix,
        input,
        q8_flags | OptimizationCpuPackedWeights,
        &packed_owner,
        ExecutionBackend::Cpu);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
        for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
        {
            check_near(packed_q8_projected.row(input_row)[matrix_row], scalar_row(matrix_row, input_row), 0.15f);
            check_near(packed_q8_projected.row(input_row)[matrix_row], unpacked_q8_projected.row(input_row)[matrix_row], 0.15f);
            check_near(owner_packed_q8_projected.row(input_row)[matrix_row], packed_q8_projected.row(input_row)[matrix_row], 0.15f);
        }
    if (mxfp4_q8_packed_kernel_available())
        check(static_cast<bool>(packed_owner.mxfp4_q8_packed));
    const std::shared_ptr<const Mxfp4Q8PackedMatrix> packed_owner_sidecar = packed_owner.mxfp4_q8_packed;
    CpuBatch owner_packed_q8_projected_again;
    linear_batch_into(
        matrix,
        input,
        owner_packed_q8_projected_again,
        q8_flags | OptimizationCpuPackedWeights,
        &packed_owner,
        ExecutionBackend::Cpu);
    check(static_cast<bool>(packed_owner.mxfp4_q8_packed == packed_owner_sidecar));
    auto vulkan_projection = NcnnVulkanMxfp4Operator::create(
        matrix,
        nullptr,
        automatic_vulkan_device_index,
        context_instance,
        g_test_optimization_flags);
    if (get_gpu_count() > 0)
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
        const CpuBatch four_row_cpu = linear_batch(
            matrix,
            four_row_input,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
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
    const CpuBatch decoded = linear_batch(
        matrix,
        decode_input,
        g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
    for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
    {
        check_near(decoded.row(0)[matrix_row], scalar_row(matrix_row, 0), 1e-5f);
    }
    TensorData odd_matrix = matrix;
    odd_matrix.shape[0] = 3;
    odd_matrix.mxfp4_blocks.resize(3 * 16);
    odd_matrix.mxfp4_scales.resize(3);
    const CpuBatch odd_projected = linear_batch(
        odd_matrix,
        input,
        g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
    check(static_cast<bool>(odd_projected.columns() == 3));
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
    {
        for (size_t matrix_row = 0; matrix_row < 3; ++matrix_row)
        {
            check_near(odd_projected.row(input_row)[matrix_row], scalar_row(matrix_row, input_row), 1e-5f);
        }
    }
    auto odd_vulkan_projection = NcnnVulkanMxfp4Operator::create(
        odd_matrix,
        nullptr,
        automatic_vulkan_device_index,
        context_instance,
        g_test_optimization_flags);
    if (get_gpu_count() > 0)
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
    CompiledOperator biased_owner;
    CpuBatch biased_owner_unpacked;
    linear_batch_into(
        matrix,
        bias,
        input,
        biased_owner_unpacked,
        q8_flags,
        &biased_owner,
        ExecutionBackend::Vulkan);
    check(static_cast<bool>(!biased_owner.mxfp4_q8_packed));
    const CpuBatch biased_owner_output = linear_batch(
        matrix,
        bias,
        input,
        q8_flags | OptimizationCpuPackedWeights,
        &biased_owner,
        ExecutionBackend::Vulkan);
    const CpuBatch biased_reference = linear_batch(
        matrix,
        bias,
        input,
        q8_flags | OptimizationCpuPackedWeights);
    for (size_t input_row = 0; input_row < input.rows(); ++input_row)
        for (size_t matrix_row = 0; matrix_row < 4; ++matrix_row)
            check_near(
                biased_owner_output.row(input_row)[matrix_row],
                biased_reference.row(input_row)[matrix_row],
                0.15f);
    if (mxfp4_q8_packed_kernel_available())
        check(static_cast<bool>(biased_owner.mxfp4_q8_packed));
    const std::shared_ptr<const Mxfp4Q8PackedMatrix> biased_owner_sidecar = biased_owner.mxfp4_q8_packed;
    CpuBatch biased_owner_output_again;
    linear_batch_into(
        matrix,
        bias,
        input,
        biased_owner_output_again,
        q8_flags | OptimizationCpuPackedWeights,
        &biased_owner,
        ExecutionBackend::Vulkan);
    check(static_cast<bool>(biased_owner.mxfp4_q8_packed == biased_owner_sidecar));
    const CpuBatch fused = fused_mxfp4_gate_up_batch(
        matrix,
        &bias,
        input,
        ExpertActivation::GptOssSwiGlu,
        7.0f,
        g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
    check(static_cast<bool>(fused.rows() == input.rows()));
    check(static_cast<bool>(fused.columns() == 2));
    for (float sigmoid_scale : {1.0f, 1.702f})
    {
        for (int step = -1000; step <= 700; ++step)
        {
            const float value = static_cast<float>(step) * 0.01f;
            const float expected = value / (1.0f + std::exp(-sigmoid_scale * value));
            check_near(approximate_scaled_silu(value, sigmoid_scale), expected, 1e-5f);
            check_near(
                scaled_silu(
                    value,
                    sigmoid_scale,
                    g_test_optimization_flags),
                expected,
                1e-5f);
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
        expert_activation_limit,
        g_test_optimization_flags);
    const CpuBatch repeated_reference = linear_batch(
        expert_down,
        expert_down_bias,
        repeated_activated,
        g_test_optimization_flags);
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
            &repeated_scratch,
            g_test_optimization_flags)));
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
    const uint64_t q8_expert_flags = g_test_optimization_flags | OptimizationCpuMxfp4Q8;
    CpuBatch q8_repeated_output;
    Mxfp4Scratch q8_repeated_scratch;
    repeated_task.output = &q8_repeated_output;
    check(static_cast<bool>(
        mxfp4_expert_batch(
            std::span<const Mxfp4Task>(&repeated_task, 1),
            &q8_repeated_scratch,
            q8_expert_flags)));
    check(q8_repeated_output.rows() == repeated_reference.rows());
    check(q8_repeated_output.columns() == repeated_reference.columns());
    for (size_t row = 0; row < q8_repeated_output.rows(); ++row)
    {
        for (uint32_t column = 0;
             column < q8_repeated_output.columns();
             ++column)
        {
            check_near(
                q8_repeated_output.row(row)[column],
                repeated_reference.row(row)[column],
                2.0f);
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
            0.0f,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const CpuBatch silu_reference = linear_batch(
            expert_down,
            silu_activated,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
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
                &silu_scratch,
                g_test_optimization_flags & ~OptimizationCpuMxfp4Q8)));
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
            &repeated_scratch,
            g_test_optimization_flags)));
    check(static_cast<bool>(
        repeated_scratch.physical_input_rows
        == std::vector<uint32_t>({4})));

    auto vulkan_expert = NcnnVulkanMxfp4ExpertOperator::create(
        expert_gate_up,
        &expert_gate_up_bias,
        expert_down,
        &expert_down_bias,
        expert_activation_limit,
        automatic_vulkan_device_index,
        ExpertActivation::GptOssSwiGlu,
        context_instance,
        g_test_optimization_flags);
    if (get_gpu_count() > 0)
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
            const CpuBatch cpu_activated = fused_mxfp4_gate_up_batch(
                expert_gate_up,
                &expert_gate_up_bias,
                expert_input,
                ExpertActivation::GptOssSwiGlu,
                expert_activation_limit,
                g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
            const CpuBatch cpu_expert = linear_batch(
                expert_down,
                expert_down_bias,
                cpu_activated,
                g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
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
        auto silu_vulkan_expert = NcnnVulkanMxfp4ExpertOperator::create(
            expert_gate_up,
            &expert_gate_up_bias,
            expert_down,
            &expert_down_bias,
            expert_activation_limit,
            automatic_vulkan_device_index,
            ExpertActivation::Silu,
            context_instance,
            g_test_optimization_flags);
        check(static_cast<bool>(silu_vulkan_expert));
        CpuBatch silu_input(1, 32);
        for (uint32_t column = 0; column < silu_input.columns(); ++column)
            silu_input.row(0)[column] = static_cast<float>(static_cast<int>((column * 11) % 23) - 11) * 0.015625f;
        const CpuBatch silu_activated = fused_mxfp4_gate_up_batch(
            expert_gate_up,
            &expert_gate_up_bias,
            silu_input,
            ExpertActivation::Silu,
            expert_activation_limit,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const CpuBatch silu_expected = linear_batch(
            expert_down,
            expert_down_bias,
            silu_activated,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        CpuBatch silu_actual;
        check(static_cast<bool>(silu_vulkan_expert->forward(silu_input, silu_actual)));
        for (uint32_t column = 0; column < silu_expected.columns(); ++column)
            check_near(silu_actual.row(0)[column], silu_expected.row(0)[column], 1e-3f);

        auto deepseek_vulkan_expert = NcnnVulkanMxfp4ExpertOperator::create(
            expert_gate_up,
            &expert_gate_up_bias,
            expert_down,
            &expert_down_bias,
            expert_activation_limit,
            automatic_vulkan_device_index,
            ExpertActivation::DeepSeekSwiGlu,
            context_instance,
            g_test_optimization_flags);
        check(static_cast<bool>(deepseek_vulkan_expert));
        CpuBatch deepseek_input(1, 32);
        for (uint32_t column = 0; column < deepseek_input.columns(); ++column)
            deepseek_input.row(0)[column] = static_cast<float>(static_cast<int>((column * 3) % 19) - 9) * 0.015625f;
        const CpuBatch deepseek_activated = fused_mxfp4_gate_up_batch(
            expert_gate_up,
            &expert_gate_up_bias,
            deepseek_input,
            ExpertActivation::DeepSeekSwiGlu,
            expert_activation_limit,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const CpuBatch deepseek_expected = linear_batch(
            expert_down,
            expert_down_bias,
            deepseek_activated,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        CpuBatch deepseek_actual;
        check(static_cast<bool>(deepseek_vulkan_expert->forward(deepseek_input, deepseek_actual)));
        for (uint32_t column = 0; column < deepseek_expected.columns(); ++column)
            check_near(deepseek_actual.row(0)[column], deepseek_expected.row(0)[column], 1e-3f);
    }
    else
    {
        check(static_cast<bool>(!vulkan_expert));
    }

    const uint64_t backend_optimization_flags = g_test_optimization_flags
                                                | OptimizationVulkanRouteAggregation;
    auto expert_backend = create_vulkan_expert_backend(
        4096,
        automatic_vulkan_device_index,
        nullptr,
        context_instance,
        backend_optimization_flags);
    if (get_gpu_count() > 0)
    {
        check(static_cast<bool>(expert_backend));
        const auto backend_gate_up = std::make_shared<TensorData>(expert_gate_up);
        const auto backend_down = std::make_shared<TensorData>(expert_down);
        expert_backend->admit("test-expert", backend_gate_up, &expert_gate_up_bias, backend_down, &expert_down_bias, 0, expert_activation_limit);
        expert_backend->admit("test-expert", backend_gate_up, &expert_gate_up_bias, backend_down, &expert_down_bias, 0, expert_activation_limit);
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
        const CpuBatch backend_activated = fused_mxfp4_gate_up_batch(
            expert_gate_up,
            &expert_gate_up_bias,
            backend_input,
            ExpertActivation::GptOssSwiGlu,
            expert_activation_limit,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const CpuBatch backend_expected = linear_batch(
            expert_down,
            expert_down_bias,
            backend_activated,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        for (uint32_t sample = 0; sample < 3; ++sample)
        {
            CpuBatch backend_output;
            check(static_cast<bool>(expert_backend->try_execute("test-expert", backend_input, backend_output) == ExpertBackendExecutionResult ::Executed));
            for (uint32_t column = 0; column < backend_expected.columns(); ++column)
            {
                check_near(backend_output.row(0)[column], backend_expected.row(0)[column], 1e-3f);
            }
        }
        auto backend_gate_up_second = std::make_shared<TensorData>(expert_gate_up);
        auto backend_down_second = std::make_shared<TensorData>(expert_down);
        backend_gate_up_second->mxfp4_blocks[0] ^= 1;
        backend_down_second->mxfp4_blocks[0] ^= 1;
        expert_backend->admit("test-expert-second", backend_gate_up_second, &expert_gate_up_bias, backend_down_second, &expert_down_bias, 0, expert_activation_limit);
        expert_backend->admit("test-expert-second", backend_gate_up_second, &expert_gate_up_bias, backend_down_second, &expert_down_bias, 0, expert_activation_limit);
        const auto second_admission_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (expert_backend->statistics().stores < 2 && std::chrono::steady_clock::now() < second_admission_deadline)
        {
            std::this_thread::yield();
        }
        check(static_cast<bool>(expert_backend->statistics().stores == 2));
        const CpuBatch backend_second_activated = fused_mxfp4_gate_up_batch(
            *backend_gate_up_second,
            &expert_gate_up_bias,
            backend_input,
            ExpertActivation::GptOssSwiGlu,
            expert_activation_limit,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        const CpuBatch backend_second_expected = linear_batch(
            *backend_down_second,
            expert_down_bias,
            backend_second_activated,
            g_test_optimization_flags & ~OptimizationCpuMxfp4Q8);
        CpuBatch backend_batch_output_first;
        CpuBatch backend_batch_output_second;
        const std::array<ExpertBackendRequest, 2> backend_requests = {{
            {
                "test-expert",
                &backend_input,
                &backend_batch_output_first,
            },
            {
                "test-expert-second",
                &backend_input,
                &backend_batch_output_second,
            },
        }};
        const auto backend_batch_start = std::chrono::steady_clock::now();
        const auto backend_batch_results = expert_backend->try_execute_batch(backend_requests);
        const uint64_t backend_batch_microseconds = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - backend_batch_start).count());
        check(static_cast<bool>(backend_batch_results.size() == 2));
        for (ExpertBackendExecutionResult result : backend_batch_results)
        {
            check(static_cast<bool>(result == ExpertBackendExecutionResult ::Executed));
        }
        for (uint32_t column = 0; column < backend_expected.columns(); ++column)
        {
            check_near(backend_batch_output_first.row(0)[column], backend_expected.row(0)[column], 1e-3f);
            check_near(backend_batch_output_second.row(0)[column], backend_second_expected.row(0)[column], 1e-3f);
        }

        const ExpertBackendStatistics backend_statistics = expert_backend->statistics();
        check(static_cast<bool>(backend_statistics.executions == 5));
        check(static_cast<bool>(backend_statistics.hits == 5));
        check(static_cast<bool>(backend_statistics.bytes_uploaded > 0));
        check(static_cast<bool>(backend_statistics.arc_frequent_size > 0));

        auto device_source = create_vulkan_victim_cache(
            4096,
            automatic_vulkan_device_index,
            context_instance,
            backend_optimization_flags);
        check(static_cast<bool>(device_source));
        device_source->admit("victim-expert", backend_gate_up, backend_down,
                             {
                                 &expert_gate_up_bias,
                                 &expert_down_bias,
                                 expert_activation_limit,
                                 true,
                             });
        device_source->wait_for_background_work();
        check(static_cast<bool>(device_source->statistics().stores == 1));
        auto source_backend = create_vulkan_expert_backend(
            0,
            automatic_vulkan_device_index,
            device_source,
            context_instance,
            backend_optimization_flags);
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

    constexpr size_t packed_matrix_rows = 13;
    std::vector<uint8_t> matrix_packed(packed_matrix_rows * test_block_count * 16);
    std::vector<uint8_t> matrix_scales(packed_matrix_rows * test_block_count);
    for (size_t row = 0; row < packed_matrix_rows; ++row)
    {
        for (uint32_t block = 0; block < test_block_count; ++block)
        {
            matrix_scales[row * test_block_count + block] = static_cast<uint8_t>(123 + ((row * 3 + block * 5) % 9));
            for (uint32_t byte = 0; byte < 16; ++byte)
            {
                const size_t offset = (row * test_block_count + block) * 16 + byte;
                matrix_packed[offset] = static_cast<uint8_t>(
                    ((row * 7 + block * 11 + byte * 3) % 16)
                    | (((row * 13 + block * 5 + byte * 9) % 16) << 4));
            }
        }
    }
    Mxfp4Q8PackedMatrix packed_matrix;
    check(static_cast<bool>(mxfp4_q8_pack_weights(
        matrix_packed.data(),
        matrix_scales.data(),
        test_block_count,
        packed_matrix_rows,
        packed_matrix)));
    check(static_cast<bool>(packed_matrix.valid()));
    check(static_cast<bool>(
        packed_matrix.storage.size()
        == mxfp4_q8_packed_storage_bytes(
            packed_matrix_rows,
            test_block_count)));

    std::array<float, 37> q8_reference_input = {};
    for (size_t index = 0; index < q8_reference_input.size(); ++index)
        q8_reference_input[index] = (static_cast<float>((index * 13) % 23) - 11.0f) * 0.37f;
    q8_reference_input[7] = 9.75f;
    q8_reference_input[34] = -4.5f;
    std::array<int8_t, 37> q8_reference_values = {};
    std::array<float, 2> q8_reference_scales = {};
    mxfp4_q8_quantize(
        q8_reference_input.data(),
        q8_reference_values.data(),
        q8_reference_scales.data(),
        static_cast<uint32_t>(q8_reference_input.size()));
    for (uint32_t block = 0; block < q8_reference_scales.size(); ++block)
    {
        const uint32_t begin = block * 32;
        const uint32_t end = std::min<uint32_t>(
            static_cast<uint32_t>(q8_reference_input.size()),
            begin + 32);
        float maximum = 0.0f;
        for (uint32_t index = begin; index < end; ++index)
            maximum = std::max(maximum, std::fabs(q8_reference_input[index]));
        const float expected_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        check_near(q8_reference_scales[block], expected_scale, 1e-7f);
        for (uint32_t index = begin; index < end; ++index)
        {
            const float normalized = std::clamp(
                q8_reference_input[index] / expected_scale,
                -127.0f,
                127.0f);
            check(q8_reference_values[index] == static_cast<int8_t>(std::lrintf(normalized)));
        }
    }

    Mxfp4Q8Batch quantized_input;
    mxfp4_q8_quantize_batch(
        strided_input.data(),
        test_input_stride,
        test_token_count,
        test_input_columns,
        quantized_input);
    std::vector<float> packed_gemm_output(test_token_count * (packed_matrix_rows + 2), -999.0f);
    mxfp4_q8_packed_gemm(
        packed_matrix,
        quantized_input.row(0),
        quantized_input.columns,
        quantized_input.row_scales(0),
        (quantized_input.columns + 31) / 32,
        test_token_count,
        packed_gemm_output.data(),
        packed_matrix_rows + 2);
    for (size_t token = 0; token < test_token_count; ++token)
    {
        for (size_t row = 0; row < packed_matrix_rows; ++row)
        {
            check_near(
                packed_gemm_output[token * (packed_matrix_rows + 2) + row],
                mxfp4_q8_dot(
                    matrix_packed.data() + row * test_block_count * 16,
                    matrix_scales.data() + row * test_block_count,
                    test_block_count,
                    quantized_input.row(token),
                    quantized_input.row_scales(token)),
                1e-4f);
        }
    }
    std::vector<float> packed_gemv_output(packed_matrix_rows, -999.0f);
    mxfp4_q8_packed_gemv(
        packed_matrix,
        quantized_input.row(1),
        quantized_input.row_scales(1),
        packed_gemv_output.data());
    for (size_t row = 0; row < packed_matrix_rows; ++row)
    {
        check_near(
            packed_gemv_output[row],
            mxfp4_q8_dot(
                matrix_packed.data() + row * test_block_count * 16,
                matrix_scales.data() + row * test_block_count,
                test_block_count,
                quantized_input.row(1),
                quantized_input.row_scales(1)),
            1e-4f);
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    check(static_cast<bool>(mxfp4_kernel_kind() == MxFp4KernelKind::ArmNeon || mxfp4_kernel_kind() == MxFp4KernelKind::ArmSve2));
#endif
    check(static_cast<bool>(std::string(mxfp4_kernel_name()).size() > 0));
    const std::string activation_kernel_name = scaled_silu_kernel_name(g_test_optimization_flags);
    check(!activation_kernel_name.empty());
}

void test_qnk_cpu_kernel()
{
    constexpr uint32_t columns = 512;
    constexpr size_t rows = 9;
    constexpr size_t input_rows = 3;
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };

    CpuBatch input(input_rows, columns);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>(column % 17) - 8) * 0.01f * static_cast<float>(row + 1);
    }

    std::vector<uint8_t> q8_row(static_cast<size_t>(qnk_storage_bytes(DType::Q8K, 1, columns)));
    qnk_q8k_quantize(input.row(0), q8_row.data(), columns);
    alignas(64) float q8_decoded[qnk_block_elements];
    qnk_dequantize_block(DType::Q8K, q8_row.data(), q8_decoded);
    float q8_scale = 0.0f;
    std::memcpy(&q8_scale, q8_row.data(), sizeof(q8_scale));
    check(std::isfinite(q8_scale));
    check(std::fabs(q8_scale) > 0.0f);
    for (uint32_t column = 0; column < qnk_block_elements; ++column)
        check(std::fabs(q8_decoded[column] - input.row(0)[column]) <= std::fabs(q8_scale) * 0.6f);
    std::vector<uint8_t> q8_batch;
    qnk_q8k_quantize_batch(input.row(0), input.columns(), input.rows(), columns, q8_batch);
    check(q8_batch.size() == input.rows() * qnk_storage_bytes(DType::Q8K, 1, columns));

    for (const DType dtype : dtypes)
    {
        const size_t block_bytes = qnk_block_bytes(dtype);
        const uint32_t block_count = columns / qnk_block_elements;
        std::vector<uint8_t> raw(static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns)));
        for (size_t index = 0; index < raw.size(); ++index)
            raw[index] = static_cast<uint8_t>((index * 37u + static_cast<uint32_t>(dtype) * 11u + 5u) & 0xffu);

        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + (row * block_count + block) * block_bytes;
                if (dtype == DType::Q2K)
                {
                    std::fill_n(encoded, 16, uint8_t{1});
                    encoded[80] = 0x00;
                    encoded[81] = 0x3c;
                    encoded[82] = 0;
                    encoded[83] = 0;
                }
                else if (dtype == DType::Q3K)
                {
                    std::fill_n(encoded + 96, 12, uint8_t{0});
                    encoded[108] = 0x00;
                    encoded[109] = 0x3c;
                }
                else if (dtype == DType::Q4K || dtype == DType::Q5K)
                {
                    encoded[0] = 0x00;
                    encoded[1] = 0x3c;
                    encoded[2] = 0;
                    encoded[3] = 0;
                    std::fill_n(encoded + 4, 4, uint8_t{1});
                    std::fill_n(encoded + 8, 4, uint8_t{0});
                }
                else if (dtype == DType::Q6K)
                {
                    std::fill_n(encoded + 192, 16, uint8_t{1});
                    encoded[208] = 0x00;
                    encoded[209] = 0x3c;
                }
                else
                {
                    const float scale = 1.0f;
                    std::memcpy(encoded, &scale, sizeof(scale));
                }
            }
        }

        QnKPack packed;
        check(qnk_pack_weights(raw.data(), raw.size(), dtype, rows, columns, packed));
        check(packed.valid());
        check(packed.block_count == block_count);
        check(packed.storage.size() == ((rows + 7) / 8) * block_count * 8 * block_bytes);
        check(packed.storage.size() == qnk_packed_storage_bytes(dtype, rows, columns));
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                const uint8_t* original = raw.data() + (row * block_count + block) * block_bytes;
                check(std::memcmp(qnk_packed_block(packed, row, block), original, block_bytes) == 0);
            }
        }

        TensorData matrix;
        matrix.dtype = dtype;
        matrix.shape = {static_cast<uint32_t>(rows), columns};
        matrix.quantized_data = raw;
        CpuBatch projected;
        linear_batch_into(matrix, input, projected, 0);
        check(projected.rows() == input_rows);
        check(projected.columns() == rows);
        CpuBatch packed_projected;
        linear_batch_into(
            matrix,
            input,
            packed_projected,
            OptimizationCpuPackedWeights);
        CompiledOperator packed_owner;
        CpuBatch owner_unpacked_projected;
        linear_batch_into(
            matrix,
            input,
            owner_unpacked_projected,
            0,
            &packed_owner,
            ExecutionBackend::Cpu);
        check(static_cast<bool>(!packed_owner.qnk_packed));
        CpuBatch owner_packed_projected;
        linear_batch_into(
            matrix,
            input,
            owner_packed_projected,
            OptimizationCpuPackedWeights,
            &packed_owner,
            ExecutionBackend::Cpu);
        check(static_cast<bool>(packed_owner.qnk_packed));
        const std::shared_ptr<const QnKPack> packed_owner_sidecar = packed_owner.qnk_packed;
        CpuBatch owner_packed_projected_again;
        linear_batch_into(
            matrix,
            input,
            owner_packed_projected_again,
            OptimizationCpuPackedWeights,
            &packed_owner,
            ExecutionBackend::Cpu);
        check(static_cast<bool>(packed_owner.qnk_packed == packed_owner_sidecar));
        check(packed_projected.rows() == input_rows);
        check(packed_projected.columns() == rows);
        check(owner_packed_projected.rows() == input_rows);
        check(owner_packed_projected.columns() == rows);
        check(owner_packed_projected_again.rows() == input_rows);
        check(owner_packed_projected_again.columns() == rows);

        if (dtype == DType::Q4K)
        {
            TensorData zero_matrix = matrix;
            std::fill(zero_matrix.quantized_data.begin(), zero_matrix.quantized_data.end(), uint8_t{0});
            CompiledOperatorTable empty_operators;
            const CompiledOperator* first_empty_owner = empty_operators.find_weight(0);
            const CompiledOperator* second_empty_owner = empty_operators.find_weight(1);
            check(first_empty_owner == nullptr);
            check(second_empty_owner == nullptr);
            check(&empty_operators.at_weight(0) == &empty_operators.at_weight(1));

            const CpuBatch matrix_without_owner = linear_batch(
                matrix, input, OptimizationCpuPackedWeights);
            const CpuBatch matrix_with_found_owner = linear_batch(
                matrix, input, OptimizationCpuPackedWeights, first_empty_owner);
            const CpuBatch zero_without_owner = linear_batch(
                zero_matrix, input, OptimizationCpuPackedWeights);
            const CpuBatch zero_with_found_owner = linear_batch(
                zero_matrix, input, OptimizationCpuPackedWeights, second_empty_owner);
            for (size_t token = 0; token < input_rows; ++token)
            {
                for (size_t row = 0; row < rows; ++row)
                {
                    check_near(
                        matrix_with_found_owner.row(token)[row],
                        matrix_without_owner.row(token)[row],
                        0.1f);
                    check_near(zero_with_found_owner.row(token)[row], 0.0f, 0.1f);
                    check_near(
                        zero_with_found_owner.row(token)[row],
                        zero_without_owner.row(token)[row],
                        0.1f);
                }
            }

            CompiledOperatorTable bound_operators;
            bound_operators.bind_weight_count(2);
            const CompiledOperator* first_bound_owner = bound_operators.find_weight(0);
            const CompiledOperator* second_bound_owner = bound_operators.find_weight(1);
            check(first_bound_owner != nullptr);
            check(second_bound_owner != nullptr);
            check(first_bound_owner != second_bound_owner);
            check(first_bound_owner == &bound_operators.at_weight(0));
            check(second_bound_owner == &bound_operators.at_weight(1));
        }

        alignas(64) float decoded[qnk_block_elements];
        for (size_t token = 0; token < input_rows; ++token)
        {
            for (size_t row = 0; row < rows; ++row)
            {
                float expected = 0.0f;
                for (uint32_t block = 0; block < block_count; ++block)
                {
                    const uint8_t* encoded = raw.data() + (row * block_count + block) * block_bytes;
                    qnk_dequantize_block(dtype, encoded, decoded);
                    for (uint32_t column = 0; column < qnk_block_elements; ++column)
                        expected += decoded[column] * input.row(token)[block * qnk_block_elements + column];
                }
                check_near(projected.row(token)[row], expected, 0.1f);
                check_near(packed_projected.row(token)[row], expected, 0.1f);
                check_near(packed_projected.row(token)[row], projected.row(token)[row], 0.1f);
                check_near(owner_packed_projected.row(token)[row], packed_projected.row(token)[row], 0.1f);
            }
        }
    }
}

void test_qnk_graph_gate_up_fusion()
{
    constexpr uint32_t vocabulary_size = 256;
    constexpr uint32_t hidden_size = 256;
    constexpr uint32_t intermediate_size = 256;
    constexpr uint32_t expert_count = 2;
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };

    for (const DType dtype : dtypes)
    {
        MoeModelDescriptor descriptor;
        descriptor.model_type = "qnk_graph_fusion_test";
        descriptor.vocabulary_size = vocabulary_size;
        descriptor.hidden_size = hidden_size;
        descriptor.intermediate_size = intermediate_size;
        descriptor.expert_count = expert_count;
        descriptor.experts_per_token = 1;
        descriptor.activation_dtype = DType::Float32;
        descriptor.kv_cache_dtype = DType::Float32;
        descriptor.norm_epsilon = 1e-5f;
        descriptor.layers.resize(1);
        descriptor.layers.front().pre_ffn_norm = NormType::RmsNorm;
        MoeDescriptor& moe = descriptor.layers.front().ffn.moe;
        moe.expert_count = expert_count;
        moe.top_k = 1;
        moe.intermediate_size = intermediate_size;
        moe.activation = ExpertActivation::Silu;
        moe.layout = ExpertLayout::GateUpDown;
        moe.expert_weight_dtype = dtype;

        WeightMapping mapping;
        auto add_float = [&mapping](const std::string& name, std::vector<uint32_t> shape) {
            TensorData tensor;
            tensor.dtype = DType::Float32;
            tensor.shape = std::move(shape);
            tensor.float32_data.assign(static_cast<size_t>(tensor.element_count()), 0.0f);
            mapping.emplace(name, std::move(tensor));
        };
        auto add_qnk = [&mapping, dtype](const std::string& name, uint32_t rows, uint32_t columns) {
            TensorData tensor;
            tensor.dtype = dtype;
            tensor.shape = {rows, columns};
            tensor.quantized_data.resize(static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns)), 0);
            mapping.emplace(name, std::move(tensor));
        };

        add_float("token_embedding.weight", {vocabulary_size, hidden_size});
        add_float("final_norm.weight", {hidden_size});
        add_float("lm_head.weight", {vocabulary_size, hidden_size});
        add_float("layers.0.pre_ffn_norm.weight", {hidden_size});
        add_float("layers.0.router.weight", {expert_count, hidden_size});
        for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id)
        {
            const std::string prefix = "layers.0.experts." + std::to_string(expert_id) + ".";
            add_qnk(prefix + "gate.weight", intermediate_size, hidden_size);
            add_qnk(prefix + "up.weight", intermediate_size, hidden_size);
            add_qnk(prefix + "down.weight", hidden_size, intermediate_size);
        }

        CompilerOption compiler_opt;
        compiler_opt.flags |= BackendVulkanDense;
        compiler_opt.optimization_flags = OptimizationDefaultFlags & ~OptimizationNcnnCpuBfloat16Linear;
        compiler_opt.num_concurrent_sessions = 3;
        auto compiled = compile_model(
            std::move(descriptor), std::move(mapping), HybridMode::CpuOnly, compiler_opt);
        check(static_cast<bool>(compiled));
        const CompiledModel& model = compiled.value();
        check(model.opt.hybrid_mode == HybridMode::CpuOnly);
        check(model.opt.vulkan_device_index == automatic_vulkan_device_index);
        check(model.opt.vulkan_device_indices.empty());
        check(model.opt.flags == 0);
        check(model.opt.optimization_flags == compiler_opt.optimization_flags);
        check(model.opt.num_concurrent_sessions == compiler_opt.num_concurrent_sessions);
        check(static_cast<bool>(model.graph.layer_plans.size() == 1));
        const ExpertPlan& expert = model.graph.layer_plans.front().moe.experts.front();
        check(static_cast<bool>(expert.gate_up_weight != invalid_tensor_handle));
        check(static_cast<bool>(expert.gate_weight == invalid_tensor_handle));
        check(static_cast<bool>(expert.up_weight == invalid_tensor_handle));
        check(static_cast<bool>(expert.layout == ExpertLayout::PackedGateUpDown));
        const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
        check(static_cast<bool>(gate_up.dtype == dtype));
        check(static_cast<bool>(gate_up.shape == std::vector<uint32_t>{intermediate_size * 2, hidden_size}));
        check(static_cast<bool>(!gate_up.qnk_interleave_rows));
        check(static_cast<bool>(gate_up.qnk_values().size() == qnk_storage_bytes(dtype, intermediate_size * 2, hidden_size)));
    }
}

void test_ncnn_vulkan_qnk_operator()
{
    constexpr uint32_t columns = 512;
    constexpr size_t rows = 5;
    constexpr size_t input_rows = 3;
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
    CpuBatch input(input_rows, columns);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((column * 7 + row * 13) % 29) - 14) * 0.03125f;
    }

    for (const DType dtype : dtypes)
    {
        const size_t block_bytes = qnk_block_bytes(dtype);
        const uint32_t block_count = columns / qnk_block_elements;
        std::vector<uint8_t> raw(static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns)));
        for (size_t index = 0; index < raw.size(); ++index)
            raw[index] = static_cast<uint8_t>((index * 37u + static_cast<uint32_t>(dtype) * 11u + 5u) & 0xffu);
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + (row * block_count + block) * block_bytes;
                if (dtype == DType::Q2K)
                {
                    std::fill_n(encoded, 16, uint8_t{1});
                    encoded[80] = 0x00;
                    encoded[81] = 0x3c;
                    encoded[82] = 0;
                    encoded[83] = 0;
                }
                else if (dtype == DType::Q3K)
                {
                    std::fill_n(encoded + 96, 12, uint8_t{0});
                    encoded[108] = 0x00;
                    encoded[109] = 0x3c;
                }
                else if (dtype == DType::Q4K || dtype == DType::Q5K)
                {
                    encoded[0] = 0x00;
                    encoded[1] = 0x3c;
                    encoded[2] = 0;
                    encoded[3] = 0;
                    std::fill_n(encoded + 4, 4, uint8_t{1});
                    std::fill_n(encoded + 8, 4, uint8_t{0});
                }
                else if (dtype == DType::Q6K)
                {
                    std::fill_n(encoded + 192, 16, uint8_t{1});
                    encoded[208] = 0x00;
                    encoded[209] = 0x3c;
                }
                else
                {
                    const float scale = 1.0f;
                    std::memcpy(encoded, &scale, sizeof(scale));
                }
            }
        }

        TensorData matrix;
        matrix.dtype = dtype;
        matrix.shape = {static_cast<uint32_t>(rows), columns};
        matrix.quantized_data = std::move(raw);
        const CpuBatch expected = linear_batch(matrix, input, 0);
        auto vulkan = NcnnVulkanQnkOperator::create(
            matrix,
            nullptr,
            automatic_vulkan_device_index,
            context_instance,
            g_test_optimization_flags | OptimizationVulkanQnK);
        if (get_gpu_count() == 0)
        {
            check(static_cast<bool>(!vulkan));
            continue;
        }
        check(static_cast<bool>(vulkan));
        check(vulkan->dtype() == dtype);
        check(vulkan->input_columns() == columns);
        check(vulkan->output_columns() == rows);
        CpuBatch actual;
        check(static_cast<bool>(vulkan->forward(input, actual)));
        check(actual.rows() == expected.rows());
        check(actual.columns() == expected.columns());
        for (size_t token = 0; token < input_rows; ++token)
        {
            for (size_t row = 0; row < rows; ++row)
                check_near(actual.row(token)[row], expected.row(token)[row], 0.2f);
        }
    }
}

void test_ncnn_vulkan_qnk_expert_operator()
{
    const uint32_t hidden_columns = 2048;
    const uint32_t intermediate_columns = 256;
    const size_t input_rows = 3;
    const uint32_t qnk_columns = hidden_columns;
    const DType dtype = DType::Q4K;
    const auto make_q4k = [](size_t rows, uint32_t columns, uint32_t seed) {
        std::vector<uint8_t> raw(static_cast<size_t>(qnk_storage_bytes(DType::Q4K, rows, columns)));
        const size_t block_bytes = qnk_block_bytes(DType::Q4K);
        const uint32_t block_count = columns / qnk_block_elements;
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + (row * block_count + block) * block_bytes;
                encoded[0] = 0x00;
                encoded[1] = 0x3c;
                encoded[2] = 0x00;
                encoded[3] = 0x3c;
                for (uint32_t index = 0; index < 8; ++index)
                    encoded[4 + index] = static_cast<uint8_t>(1u + ((seed + row + block + index) & 1u));
                for (uint32_t index = 0; index < 64; ++index)
                    encoded[16 + index] = static_cast<uint8_t>(0x11u + ((seed + row + block + index) & 0x22u));
            }
        }
        return raw;
    };

    TensorData gate_up;
    gate_up.dtype = dtype;
    gate_up.shape = {intermediate_columns * 2, qnk_columns};
    gate_up.quantized_data = make_q4k(gate_up.shape[0], qnk_columns, 7);
    TensorData down;
    down.dtype = dtype;
    down.shape = {hidden_columns, intermediate_columns};
    down.quantized_data = make_q4k(down.shape[0], intermediate_columns, 19);

    CpuBatch input(input_rows, hidden_columns);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((row * 17 + column * 5) % 31) - 15) * 0.015625f;
    }

    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
    const std::array<ExpertActivation, 3> activations = {
        ExpertActivation::Silu,
        ExpertActivation::DeepSeekSwiGlu,
        ExpertActivation::GptOssSwiGlu,
    };
    for (const ExpertActivation activation : activations)
    {
        const CpuBatch gate_up_output = linear_batch(gate_up, input, 0);
        CpuBatch activated(input_rows, intermediate_columns);
        for (size_t row = 0; row < input_rows; ++row)
        {
            for (uint32_t column = 0; column < intermediate_columns; ++column)
            {
                const float gate = gate_up_output.row(row)[column * 2];
                const float up = gate_up_output.row(row)[column * 2 + 1];
                if (activation == ExpertActivation::GptOssSwiGlu)
                {
                    const float silu = gate / (1.0f + std::exp(-1.702f * gate));
                    activated.row(row)[column] = silu * (up + 1.0f);
                }
                else
                {
                    const float silu = gate / (1.0f + std::exp(-gate));
                    activated.row(row)[column] = silu * up;
                }
            }
        }
        const CpuBatch expected = linear_batch(down, activated, 0);
        auto vulkan = NcnnVulkanQnkExpertOperator::create(
            gate_up,
            nullptr,
            down,
            nullptr,
            0.0f,
            automatic_vulkan_device_index,
            activation,
            context_instance,
            g_test_optimization_flags | OptimizationVulkanQnK);
        if (get_gpu_count() == 0)
        {
            check(static_cast<bool>(!vulkan));
            continue;
        }
        check(static_cast<bool>(vulkan));
        CpuBatch actual;
        check(static_cast<bool>(vulkan->forward(input, actual)));
        check(actual.rows() == expected.rows());
        check(actual.columns() == expected.columns());
        for (size_t row = 0; row < actual.rows(); ++row)
        {
            for (uint32_t column = 0; column < actual.columns(); ++column)
                check_near(actual.row(row)[column], expected.row(row)[column], 0.2f);
        }
    }

    TensorData packed_gate_up = gate_up;
    packed_gate_up.qnk_interleave_rows = false;
    const size_t qnk_block_bytes_value = qnk_block_bytes(dtype);
    const uint32_t qnk_block_count = qnk_columns / qnk_block_elements;
    const uint32_t packed_intermediate_rows = intermediate_columns;
    for (uint32_t packed_row = 0; packed_row < packed_intermediate_rows * 2; ++packed_row)
    {
        const uint32_t source_row = packed_row < packed_intermediate_rows
                                        ? packed_row * 2
                                        : (packed_row - packed_intermediate_rows) * 2 + 1;
        for (uint32_t block = 0; block < qnk_block_count; ++block)
        {
            const size_t destination_offset = (static_cast<size_t>(packed_row) * qnk_block_count + block) * qnk_block_bytes_value;
            const size_t source_offset = (static_cast<size_t>(source_row) * qnk_block_count + block) * qnk_block_bytes_value;
            std::memcpy(
                packed_gate_up.quantized_data.data() + destination_offset,
                gate_up.quantized_data.data() + source_offset,
                qnk_block_bytes_value);
        }
    }
    const CpuBatch packed_gate_output = linear_batch(packed_gate_up, input, 0);
    CpuBatch packed_activated(input_rows, intermediate_columns);
    for (size_t row = 0; row < input_rows; ++row)
    {
        for (uint32_t column = 0; column < intermediate_columns; ++column)
        {
            const float gate = packed_gate_output.row(row)[column];
            const float up = packed_gate_output.row(row)[intermediate_columns + column];
            const float silu = gate / (1.0f + std::exp(-1.702f * gate));
            packed_activated.row(row)[column] = silu * (up + 1.0f);
        }
    }
    const CpuBatch packed_expected = linear_batch(down, packed_activated, 0);
    auto packed_vulkan = NcnnVulkanQnkExpertOperator::create(
        packed_gate_up,
        nullptr,
        down,
        nullptr,
        0.0f,
        automatic_vulkan_device_index,
        ExpertActivation::GptOssSwiGlu,
        context_instance,
        g_test_optimization_flags | OptimizationVulkanQnK);
    if (get_gpu_count() == 0)
    {
        check(static_cast<bool>(!packed_vulkan));
    }
    else
    {
        check(static_cast<bool>(packed_vulkan));
        CpuBatch packed_actual;
        check(static_cast<bool>(packed_vulkan->forward(input, packed_actual)));
        check(packed_actual.rows() == packed_expected.rows());
        check(packed_actual.columns() == packed_expected.columns());
        for (size_t row = 0; row < packed_actual.rows(); ++row)
        {
            for (uint32_t column = 0; column < packed_actual.columns(); ++column)
            {
                check_near(packed_actual.row(row)[column], packed_expected.row(row)[column], 0.2f);
            }
        }
    }

    if (get_gpu_count() == 0)
        return;
    CpuBatch gate_up_output = linear_batch(gate_up, input, 0);
    CpuBatch activated(input_rows, intermediate_columns);
    for (size_t row = 0; row < input_rows; ++row)
    {
        for (uint32_t column = 0; column < intermediate_columns; ++column)
        {
            const float gate = gate_up_output.row(row)[column * 2];
            const float up = gate_up_output.row(row)[column * 2 + 1];
            const float silu = gate / (1.0f + std::exp(-1.702f * gate));
            activated.row(row)[column] = silu * (up + 1.0f);
        }
    }
    const CpuBatch expected = linear_batch(down, activated, 0);
    const uint64_t qnk_pair_bytes = gate_up.quantized_data.size() + down.quantized_data.size();
    const uint64_t backend_flags = g_test_optimization_flags
                                   | OptimizationVulkanQnK;
    const auto backend = create_vulkan_expert_backend(
        qnk_pair_bytes * 2 + 4096,
        automatic_vulkan_device_index,
        nullptr,
        context_instance,
        backend_flags);
    check(static_cast<bool>(backend));
    const auto backend_gate_up = std::make_shared<TensorData>(gate_up);
    const auto backend_down = std::make_shared<TensorData>(down);
    backend->admit(
        "qnk-expert",
        backend_gate_up,
        nullptr,
        backend_down,
        nullptr,
        0,
        0.0f,
        ExpertActivation::GptOssSwiGlu);
    backend->wait_for_background_work();
    check(backend->statistics().stores == 1);
    CpuBatch actual;
    check(backend->try_execute("qnk-expert", input, actual) == ExpertBackendExecutionResult::Executed);
    check(actual.rows() == expected.rows());
    check(actual.columns() == expected.columns());
    for (size_t row = 0; row < actual.rows(); ++row)
    {
        for (uint32_t column = 0; column < actual.columns(); ++column)
            check_near(actual.row(row)[column], expected.row(row)[column], 0.2f);
    }

    auto backend_gate_up_second = std::make_shared<TensorData>(gate_up);
    auto backend_down_second = std::make_shared<TensorData>(down);
    const CpuBatch second_gate_up_output = linear_batch(*backend_gate_up_second, input, 0);
    CpuBatch second_activated(input_rows, intermediate_columns);
    for (size_t row = 0; row < input_rows; ++row)
    {
        for (uint32_t column = 0; column < intermediate_columns; ++column)
        {
            const float gate = second_gate_up_output.row(row)[column * 2];
            const float up = second_gate_up_output.row(row)[column * 2 + 1];
            const float silu = gate / (1.0f + std::exp(-1.702f * gate));
            second_activated.row(row)[column] = silu * (up + 1.0f);
        }
    }
    const CpuBatch second_expected = linear_batch(*backend_down_second, second_activated, 0);
    auto direct_second = NcnnVulkanQnkExpertOperator::create(
        *backend_gate_up_second,
        nullptr,
        *backend_down_second,
        nullptr,
        0.0f,
        automatic_vulkan_device_index,
        ExpertActivation::GptOssSwiGlu,
        context_instance,
        g_test_optimization_flags | OptimizationVulkanQnK);
    CpuBatch direct_second_output;
    check(static_cast<bool>(direct_second));
    check(static_cast<bool>(direct_second->forward(input, direct_second_output)));
    backend->admit(
        "qnk-expert-second",
        backend_gate_up_second,
        nullptr,
        backend_down_second,
        nullptr,
        0,
        0.0f,
        ExpertActivation::GptOssSwiGlu);
    backend->wait_for_background_work();
    check(backend->statistics().stores == 2);

    CpuBatch second_single_output;
    check(backend->try_execute("qnk-expert-second", input, second_single_output) == ExpertBackendExecutionResult::Executed);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < expected.columns(); ++column)
        {
            check_near(second_single_output.row(row)[column], second_expected.row(row)[column], 0.2f);
        }
    }

    CpuBatch batch_output_first;
    CpuBatch batch_output_second;
    const std::array<ExpertBackendRequest, 2> batch_requests = {{
        {"qnk-expert", &input, &batch_output_first},
        {"qnk-expert-second", &input, &batch_output_second},
    }};
    const auto batch_results = backend->try_execute_batch(batch_requests);
    check(batch_results.size() == batch_requests.size());
    check(batch_results[0] == ExpertBackendExecutionResult::Executed);
    check(batch_results[1] == ExpertBackendExecutionResult::Executed);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < expected.columns(); ++column)
        {
            check_near(batch_output_first.row(row)[column], expected.row(row)[column], 0.2f);
            check_near(batch_output_second.row(row)[column], second_expected.row(row)[column], 0.2f);
        }
    }

    ActivationBuffer aggregated_output(2, expected.columns());
    std::vector<ExpertRoute> first_routes = {
        {0, 0, 0.15f},
        {0, 1, 0.25f},
        {1, 2, 0.10f},
    };
    std::vector<ExpertRoute> second_routes = {
        {0, 0, 0.20f},
        {1, 1, 0.15f},
        {1, 2, 0.15f},
    };
    CpuBatch aggregation_output_first(1, 1);
    CpuBatch aggregation_output_second(1, 1);
    aggregation_output_first.row(0)[0] = 71.0f;
    aggregation_output_second.row(0)[0] = 72.0f;
    std::fill_n(aggregated_output.row(0), aggregated_output.rows() * aggregated_output.columns(), 123.0f);
    uint8_t first_completed = 0;
    uint8_t second_completed = 0;
    std::array<ExpertBackendRequest, 2> aggregation_requests = {{
        {"qnk-expert", &input, &aggregation_output_first},
        {"qnk-expert-second", &input, &aggregation_output_second},
    }};
    aggregation_requests[0].route_aggregation = {
        &aggregated_output,
        first_routes,
        2,
        &first_completed,
        true,
    };
    aggregation_requests[1].route_aggregation = {
        &aggregated_output,
        second_routes,
        2,
        &second_completed,
        true,
    };
    const auto check_unpublished = [&] {
        check(first_completed == 0);
        check(second_completed == 0);
        check(aggregation_output_first.rows() == 1 && aggregation_output_first.columns() == 1);
        check(aggregation_output_second.rows() == 1 && aggregation_output_second.columns() == 1);
        check(aggregation_output_first.row(0)[0] == 71.0f);
        check(aggregation_output_second.row(0)[0] == 72.0f);
        check(aggregated_output.rows() == 2 && aggregated_output.columns() == expected.columns());
        check(std::all_of(aggregated_output.values().begin(), aggregated_output.values().end(), [](float value) { return value == 123.0f; }));
    };

    // Aborting before or after wait must leave every caller buffer untouched.
    for (bool abort_before_wait : {false, true})
    {
        auto submission = backend->submit_batch(aggregation_requests);
        check(static_cast<bool>(submission));
        if (abort_before_wait)
            submission->abort();
        const auto results = submission->wait();
        check(results.size() == aggregation_requests.size());
        check(results[0] == ExpertBackendExecutionResult::Executed);
        check(results[1] == ExpertBackendExecutionResult::Executed);
        check_unpublished();
        submission->abort();
        submission->abort();
        check(!submission->commit());
        check_unpublished();
    }

    // Conflicting aggregate destinations must reject the entire publication.
    {
        ActivationBuffer other_aggregate(2, expected.columns());
        std::fill_n(other_aggregate.row(0), other_aggregate.rows() * other_aggregate.columns(), 456.0f);
        auto invalid_requests = aggregation_requests;
        invalid_requests[1].route_aggregation.output = &other_aggregate;
        auto submission = backend->submit_batch(invalid_requests);
        check(static_cast<bool>(submission));
        const auto results = submission->wait();
        check(results.size() == invalid_requests.size());
        check(results[0] == ExpertBackendExecutionResult::Executed);
        check(results[1] == ExpertBackendExecutionResult::Executed);
        check_unpublished();
        check(!submission->commit());
        check_unpublished();
        check(other_aggregate.rows() == 2 && other_aggregate.columns() == expected.columns());
        check(std::all_of(other_aggregate.values().begin(), other_aggregate.values().end(), [](float value) { return value == 456.0f; }));
        submission->abort();
        submission->abort();
        check(!submission->commit());
        check_unpublished();
    }

    auto aggregation_submission = backend->submit_batch(aggregation_requests);
    check(static_cast<bool>(aggregation_submission));
    const auto aggregation_results = aggregation_submission->wait();
    check_unpublished();
    check(aggregation_submission->commit());
    check(aggregation_results.size() == aggregation_requests.size());
    check(aggregation_results[0] == ExpertBackendExecutionResult::Executed);
    check(aggregation_results[1] == ExpertBackendExecutionResult::Executed);
    check(first_completed == 1);
    check(second_completed == 1);
    for (size_t token = 0; token < aggregated_output.rows(); ++token)
    {
        for (uint32_t column = 0; column < expected.columns(); ++column)
        {
            float expected_aggregation = 0.0f;
            for (size_t row = 0; row < input.rows(); ++row)
            {
                if (first_routes[row].token_index == token)
                    expected_aggregation += first_routes[row].weight * expected.row(row)[column];
                if (second_routes[row].token_index == token)
                    expected_aggregation += second_routes[row].weight * second_expected.row(row)[column];
            }
            check_near(aggregated_output.row(token)[column], expected_aggregation, 0.2f);
        }
    }
    check(backend->statistics().route_aggregation_batches >= 1);

    const uint64_t aggregation_batches = backend->statistics().route_aggregation_batches;
    std::fill_n(
        aggregated_output.row(0),
        aggregated_output.rows() * aggregated_output.columns(),
        123.0f);
    CpuBatch partial_output_first;
    CpuBatch partial_output_second;
    CpuBatch missing_output(input.rows(), expected.columns());
    std::fill_n(
        missing_output.row(0),
        missing_output.rows() * missing_output.columns(),
        321.0f);
    uint8_t partial_first_completed = 0;
    uint8_t partial_second_completed = 0;
    uint8_t missing_completed = 0;
    std::array<ExpertBackendRequest, 3> partial_requests = {{
        {"qnk-expert", &input, &partial_output_first},
        {"qnk-expert-second", &input, &partial_output_second},
        {"qnk-expert-missing", &input, &missing_output},
    }};
    partial_requests[0].route_aggregation = {
        &aggregated_output,
        first_routes,
        2,
        &partial_first_completed,
        false,
    };
    partial_requests[1].route_aggregation = {
        &aggregated_output,
        second_routes,
        2,
        &partial_second_completed,
        false,
    };
    partial_requests[2].route_aggregation = {
        &aggregated_output,
        first_routes,
        2,
        &missing_completed,
        true,
    };
    auto partial_submission = backend->submit_batch(partial_requests);
    check(static_cast<bool>(partial_submission));
    const auto partial_results = partial_submission->wait();
    check(partial_submission->commit());
    check(partial_results.size() == partial_requests.size());
    check(partial_results[0] == ExpertBackendExecutionResult::Executed);
    check(partial_results[1] == ExpertBackendExecutionResult::Executed);
    check(partial_results[2] == ExpertBackendExecutionResult::NotResident);
    check(partial_first_completed == 0);
    check(partial_second_completed == 0);
    check(missing_completed == 0);
    check(aggregated_output.row(0)[0] == 123.0f);
    check(missing_output.row(0)[0] == 321.0f);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < expected.columns(); ++column)
        {
            check_near(partial_output_first.row(row)[column], expected.row(row)[column], 0.2f);
            check_near(partial_output_second.row(row)[column], second_expected.row(row)[column], 0.2f);
        }
    }
    check(backend->statistics().route_aggregation_batches
          == aggregation_batches);
}

class TestExpertVictimCache final : public ExpertVictimCache
{
public:
    explicit TestExpertVictimCache(uint64_t _capacity)
        : cache_size(_capacity)
    {
    }

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, std::shared_ptr<const TensorData> down, ExpertVictimExecutionMetadata) override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        ExpertVictimPair pair;
        pair.gate_up = std::make_shared<TensorData>(*gate_up);
        pair.down = std::make_shared<TensorData>(*down);
        entries_[std::move(key)] = std::move(pair);
        ++statistics_.admissions;
        ++statistics_.stores;
        statistics_.resident_size += gate_up->mxfp4_blocks.size() + gate_up->mxfp4_scales.size() + down->mxfp4_blocks.size() + down->mxfp4_scales.size();
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

    uint64_t capacity() const noexcept override
    {
        return cache_size;
    }

private:
    uint64_t cache_size = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ExpertVictimPair> entries_;
    ExpertVictimCacheStatistics statistics_;
};

class DelayedExpertVictimCache final : public ExpertVictimCache
{
public:
    void admit(std::string, std::shared_ptr<const TensorData>, std::shared_ptr<const TensorData>, ExpertVictimExecutionMetadata) override
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

    uint64_t capacity() const noexcept override
    {
        return 68;
    }

private:
    static std::shared_ptr<TensorData> make_tensor(const TensorData& source, uint8_t value)
    {
        auto result = std::make_shared<TensorData>(source);
        result->mxfp4_file_storage.reset();
        result->mxfp4_blocks.resize(source.mxfp4_file_storage->blocks_size);
        std::fill_n(result->mxfp4_blocks.data(), result->mxfp4_blocks.size(), value);
        result->mxfp4_scales.resize(source.mxfp4_file_storage->scales_size);
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
    check(static_cast<bool>(sharded->capacity() == 102));

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

        auto shared_data = mapped.value()->share_data();
        const std::weak_ptr<MappedFileRange> mapped_lifetime = mapped.value();
        auto independent = MappedFileRange::open(path, offset + 1, byte_count - 2);
        check(static_cast<bool>(independent));
        check(independent.value().get() != mapped.value().get());
        auto independent_data = independent.value()->share_data();
        const std::weak_ptr<MappedFileRange> independent_lifetime = independent.value();

        mapped.value().reset();
        independent.value().reset();
        check(!mapped_lifetime.expired());
        check(!independent_lifetime.expired());
        check(shared.front() == expected[offset]);
        check(shared.back() == expected[offset + byte_count - 1]);
        check(shared_data.get()[0] == expected[offset]);
        check(shared_data.get()[byte_count - 1] == expected[offset + byte_count - 1]);

        shared_data.reset();
        check(!mapped_lifetime.expired());
        check(shared.front() == expected[offset]);
        check(shared.back() == expected[offset + byte_count - 1]);
        shared.resize(0);
        check(mapped_lifetime.expired());
        check(!independent_lifetime.expired());
        check(independent_data.get()[0] == expected[offset + 1]);
        check(independent_data.get()[byte_count - 3] == expected[offset + byte_count - 2]);
        independent_data.reset();
        check(independent_lifetime.expired());

        auto truncated = MappedFileRange::open(path, file_size - 8, 16);
        check(static_cast<bool>(!truncated));
        check(static_cast<bool>(truncated.error().code == ErrorCode::InvalidModel));
        auto empty = MappedFileRange::open(path, offset, 0);
        check(!empty);
        check(empty.error().code == ErrorCode::InvalidArgument);
        auto missing = MappedFileRange::open(directory / "missing.bin", 0, 1);
        check(!missing);
        check(missing.error().code == ErrorCode::IoError);
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
    check(static_cast<bool>(storage.blocks_offset + storage.blocks_size == storage.scales_offset));

    ExpertCache cache(34, 1, {}, ExpertCacheBufferedReads);
    auto pair = cache.acquire_pair(expert.value(), expert.value());
    check(static_cast<bool>(pair));
    check(static_cast<bool>(pair.value().gate_up->mxfp4_blocks.front() == 9));
    check(static_cast<bool>(pair.value().gate_up->mxfp4_scales.front() == 123));

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_safetensors_qnk_source_binding()
{
    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_safetensors_qnk_packed_test_");
    const auto write_archive = [](const std::filesystem::path& path, std::string header, const std::vector<uint8_t>& data) {
        while (header.size() % 8 != 0)
            header.push_back(' ');
        const uint64_t header_size = header.size();
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
        stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    };
    constexpr uint32_t expert_count = 2;
    constexpr uint32_t rows = 2;
    constexpr uint32_t columns = 256;
    const size_t expert_bytes = static_cast<size_t>(qnk_storage_bytes(DType::Q4K, rows, columns));
    std::vector<uint8_t> source(expert_count * expert_bytes + expert_bytes, 3);
    write_archive(
        directory / "model.safetensors",
        R"({"experts":{"dtype":"U8","shape":[2,2,1,144],"data_offsets":[0,576]},"matrix":{"dtype":"U8","shape":[2,1,144],"data_offsets":[576,864]}})",
        source);
    std::vector<uint8_t> packed(expert_bytes * 2, 91);
    write_archive(
        directory / "ncnn-moe-packed-qnk.safetensors",
        R"({"__ncnn_moe_packed__.1.experts":{"dtype":"U8","shape":[2,1,144],"data_offsets":[0,288]},"__ncnn_moe_packed__.matrix":{"dtype":"U8","shape":[2,1,144],"data_offsets":[288,576]}})",
        packed);

    auto archive = SafetensorsArchive::open(directory);
    check(static_cast<bool>(archive));
    const auto detected_dtype = archive.value().find_qnk_expert_dtype(
        "experts",
        expert_count,
        rows,
        columns);
    check(static_cast<bool>(detected_dtype && detected_dtype.value() == DType::Q4K));
    auto expert = archive.value().load_qnk_expert("experts", DType::Q4K, 1, expert_count, rows, columns);
    check(static_cast<bool>(expert));
    check(expert.value().dtype == DType::Q4K);
    check(expert.value().shape == std::vector<uint32_t>{rows, columns});
    check(expert.value().qnk_values().size() == expert_bytes);
    check(expert.value().qnk_values().front() == 3);
    check(expert.value().qnk_values().back() == 3);
    auto matrix = archive.value().load_qnk_tensor("matrix", DType::Q4K, rows, columns);
    check(static_cast<bool>(matrix));
    check(matrix.value().qnk_values().size() == expert_bytes);
    check(matrix.value().qnk_values().front() == 3);

    const std::filesystem::path sidecar_only_directory = create_unique_test_directory("ncnn_moe_safetensors_qnk_sidecar_only_test_");
    write_archive(
        sidecar_only_directory / "ncnn-moe-packed-qnk.safetensors",
        R"({"__ncnn_moe_packed__.0.experts":{"dtype":"U8","shape":[2,1,144],"data_offsets":[0,288]},"__ncnn_moe_packed__.1.experts":{"dtype":"U8","shape":[2,1,144],"data_offsets":[288,576]}})",
        packed);
    auto sidecar_only = SafetensorsArchive::open(sidecar_only_directory);
    check(static_cast<bool>(sidecar_only));
    check(!sidecar_only.value().find_qnk_expert_dtype("experts", expert_count, rows, columns));
    check(!sidecar_only.value().load_qnk_expert("experts", DType::Q4K, 0, expert_count, rows, columns));

    const std::filesystem::path bfloat16_directory = create_unique_test_directory("ncnn_moe_safetensors_qnk_bfloat16_sidecar_test_");
    write_archive(
        bfloat16_directory / "model.safetensors",
        R"({"experts":{"dtype":"BF16","shape":[2,2,256],"data_offsets":[0,2048]}})",
        std::vector<uint8_t>(2048, 7));
    write_archive(
        bfloat16_directory / "ncnn-moe-packed-qnk.safetensors",
        R"({"__ncnn_moe_packed__.0.experts":{"dtype":"U8","shape":[2,1,144],"data_offsets":[0,288]},"__ncnn_moe_packed__.1.experts":{"dtype":"U8","shape":[2,1,144],"data_offsets":[288,576]}})",
        packed);
    auto bfloat16_with_sidecar = SafetensorsArchive::open(bfloat16_directory);
    check(static_cast<bool>(bfloat16_with_sidecar));
    check(!bfloat16_with_sidecar.value().find_qnk_expert_dtype("experts", expert_count, rows, columns));
    check(!bfloat16_with_sidecar.value().load_qnk_expert("experts", DType::Q4K, 0, expert_count, rows, columns));

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    std::filesystem::remove_all(sidecar_only_directory, ignored);
    std::filesystem::remove_all(bfloat16_directory, ignored);
}

void test_file_backed_bfloat16_expert_cache()
{
    const auto mapped_bfloat16 = [](
                                     std::vector<uint32_t> shape,
                                     std::initializer_list<float> source_values) {
        auto storage = std::make_shared<std::vector<uint16_t>>();
        storage->reserve(source_values.size());
        for (float value : source_values)
            storage->push_back(float_to_bfloat16(value));

        TensorData tensor;
        tensor.dtype = DType::BFloat16;
        tensor.shape = std::move(shape);
        tensor.mapped_data = std::shared_ptr<const uint8_t>(
            storage,
            reinterpret_cast<const uint8_t*>(storage->data()));
        tensor.mapped_size = storage->size() * sizeof(uint16_t);
        return tensor;
    };

    const TensorData gate_up = mapped_bfloat16(
        {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const TensorData down = mapped_bfloat16(
        {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    constexpr uint64_t pair_size = 16;

    ExpertCache mapped_cache(
        pair_size, 0, {}, ExpertCacheMemoryMapRanges);
    auto mapped = mapped_cache.acquire_pair(gate_up, down);
    check(static_cast<bool>(mapped));
    check(!mapped.value().cache_hit);
    check(mapped.value().bytes_read == pair_size);
    check(static_cast<bool>(mapped.value().gate_up->mapped_data));
    check(mapped.value().gate_up->bfloat16_data.empty());
    check_near(
        bfloat16_to_float(mapped.value().gate_up->bfloat16_values()[2]),
        3.0f, 0.0f);
    const std::string key = ExpertCache::make_pair_key(gate_up, down);
    check(mapped_cache.is_ready(gate_up, down, key));
    auto hit = mapped_cache.acquire_pair(
        gate_up, down, std::numeric_limits<uint32_t>::max(), key);
    check(static_cast<bool>(hit));
    check(hit.value().cache_hit);
    check(hit.value().bytes_read == 0);
    const ExpertCacheStatistics mapped_statistics = mapped_cache.statistics();
    check(mapped_statistics.misses == 1);
    check(mapped_statistics.hits == 1);
    check(mapped_statistics.resident_size == pair_size);
    check(mapped_statistics.mapped_ranges == 2);
    check(mapped_statistics.mapped_bytes == pair_size);

    ExpertCache copied_cache(pair_size);
    auto copied = copied_cache.acquire_pair(gate_up, down);
    check(static_cast<bool>(copied));
    check(!copied.value().gate_up->mapped_data);
    check(copied.value().gate_up->bfloat16_data.size() == 4);
    check_near(
        bfloat16_to_float(copied.value().down->bfloat16_values()[3]),
        8.0f, 0.0f);

    const TensorData gate_up_two = mapped_bfloat16(
        {2, 2}, {9.0f, 10.0f, 11.0f, 12.0f});
    const TensorData down_two = mapped_bfloat16(
        {2, 2}, {13.0f, 14.0f, 15.0f, 16.0f});
    const TensorData gate_up_three = mapped_bfloat16(
        {2, 2}, {17.0f, 18.0f, 19.0f, 20.0f});
    const TensorData down_three = mapped_bfloat16(
        {2, 2}, {21.0f, 22.0f, 23.0f, 24.0f});
    const std::array<ExpertCachePairRequest, 3> requests = {{
        {&gate_up, &down, 0, ExpertCache::make_pair_key(gate_up, down)},
        {&gate_up_two, &down_two, 0, ExpertCache::make_pair_key(gate_up_two, down_two)},
        {&gate_up_three, &down_three, 0, ExpertCache::make_pair_key(gate_up_three, down_three)},
    }};
    ExpertCache bounded_cache(pair_size * 2);
    std::array<uint8_t, 3> acquired_pairs{};
    size_t acquired_count = 0;
    while (acquired_count != requests.size())
    {
        std::array<ExpertCachePairRequest, 3> pending_requests{};
        std::array<size_t, 3> pending_indices{};
        size_t pending_count = 0;
        for (size_t index = 0; index < requests.size(); ++index)
        {
            if (acquired_pairs[index] != 0)
                continue;
            pending_requests[pending_count] = requests[index];
            pending_indices[pending_count] = index;
            ++pending_count;
        }
        std::array<ExpertCacheLease, 3> leases;
        auto ready = bounded_cache.wait_acquire_ready_pairs(
            std::span<const ExpertCachePairRequest>(
                pending_requests.data(), pending_count),
            std::span<ExpertCacheLease>(leases.data(), pending_count));
        check(static_cast<bool>(ready));
        check(ready.value() != 0);
        for (size_t index = 0; index < pending_count; ++index)
        {
            if (!leases[index].gate_up)
                continue;
            acquired_pairs[pending_indices[index]] = 1;
            ++acquired_count;
        }
    }
    check(bounded_cache.statistics().resident_size <= pair_size * 2);
}

void test_file_backed_mxfp4_expert_cache()
{
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
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

    auto file_backed = [&](uint64_t block_offset, uint64_t scale_offset, uint64_t block_size = 16) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {1, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = blocks_path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_size = block_size;
        storage->scales_path = scales_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_size = 1;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };

    const TensorData gate_zero = file_backed(0, 0);
    const TensorData down_zero = file_backed(16, 1);
    const TensorData gate_one = file_backed(32, 2);
    const TensorData down_one = file_backed(48, 3);
    ExpertCache cache(34, 0, {}, ExpertCacheMemoryMapRanges);
    const CompiledOperator* first_gate_up_operator = nullptr;
    const CompiledOperator* first_down_operator = nullptr;
    {
        auto first = cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(first));
        check(static_cast<bool>(!first.value().cache_hit));
        check(static_cast<bool>(first.value().bytes_read == 34));
        check(static_cast<bool>(first.value().gate_up->mxfp4_blocks.front() == 0));
        check(static_cast<bool>(first.value().down->mxfp4_blocks.front() == 16));
        check(static_cast<bool>(first.value().gate_up->mxfp4_scales.front() == 101));
        check(static_cast<bool>(first.value().down->mxfp4_scales.front() == 102));
        check(static_cast<bool>(first.value().gate_up_operator != nullptr));
        check(static_cast<bool>(first.value().down_operator != nullptr));
        first_gate_up_operator = first.value().gate_up_operator;
        first_down_operator = first.value().down_operator;
    }
    {
        const std::string prepared_key = ExpertCache::make_pair_key(gate_zero, down_zero);
        check(cache.is_ready(gate_zero, down_zero, prepared_key));
        auto hit = cache.acquire_pair(gate_zero, down_zero, std::numeric_limits<uint32_t>::max(), prepared_key);
        check(static_cast<bool>(hit));
        check(static_cast<bool>(hit.value().cache_hit));
        check(static_cast<bool>(hit.value().bytes_read == 0));
        check(static_cast<bool>(hit.value().gate_up_operator == first_gate_up_operator));
        check(static_cast<bool>(hit.value().down_operator == first_down_operator));
    }
    {
        const std::string prepared_key = ExpertCache::make_pair_key(gate_zero, down_zero);
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
        check(static_cast<bool>(leases[0].gate_up_operator == first_gate_up_operator));
        check(static_cast<bool>(leases[0].down_operator == first_down_operator));
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
    check(static_cast<bool>(statistics.resident_size == 34));
    check(static_cast<bool>(statistics.mapped_ranges == 8));
    check(static_cast<bool>(statistics.mapped_bytes == statistics.bytes_read));
    check(static_cast<bool>(statistics.num_io_threads > 0));
    check(static_cast<bool>(statistics.num_active_io_threads == statistics.num_io_threads));

    ExpertCacheLease retained_lease;
    {
        ExpertCache pin_cache(34, 1, {}, ExpertCacheBufferedReads);
        auto acquired = pin_cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value().gate_up_operator != nullptr));
        check(static_cast<bool>(acquired.value().down_operator != nullptr));
        ExpertCacheLease copied = acquired.value();
        retained_lease = std::move(copied);
    }
    check(static_cast<bool>(retained_lease.gate_up_operator != nullptr));
    check(static_cast<bool>(retained_lease.down_operator != nullptr));
    check(static_cast<bool>(retained_lease.gate_up_operator->mxfp4_q8_packed == nullptr));
    check(static_cast<bool>(retained_lease.down_operator->mxfp4_q8_packed == nullptr));

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
        storage->blocks_size = 16;
        storage->scales_path = packed_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_size = 1;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    const TensorData packed_gate = packed_tensor(0, 16);
    const TensorData packed_down = packed_tensor(17, 33);
    ExpertCache packed_cache(34, 1, {}, ExpertCacheBufferedReads);
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
        storage->blocks_size = 16;
        storage->scales_path = coalesced_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_size = 1;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    const TensorData coalesced_gate_zero = coalesced_tensor(0, 16);
    const TensorData coalesced_down_zero = coalesced_tensor(17, 33);
    const TensorData coalesced_gate_one = coalesced_tensor(34, 50);
    const TensorData coalesced_down_one = coalesced_tensor(51, 67);
    ExpertCache coalesced_cache(
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

    ExpertCache resolved_predictions(68, 1, {}, ExpertCacheBufferedReads);
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

    ExpertCache forward_aware(
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

    ExpertCache predicted_protection(
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
    clustered_gate_storage->blocks_size = 32;
    clustered_gate_storage->scales_path = clustered_path.string();
    clustered_gate_storage->scales_offset = 0;
    clustered_gate_storage->scales_size = 2;
    clustered_gate_storage->secondary_blocks_path = clustered_path.string();
    clustered_gate_storage->secondary_blocks_offset = 80;
    clustered_gate_storage->secondary_blocks_size = 32;
    clustered_gate_storage->secondary_scales_path = clustered_path.string();
    clustered_gate_storage->secondary_scales_offset = 4;
    clustered_gate_storage->secondary_scales_size = 2;
    clustered_gate_storage->interleave_rows = true;
    clustered_gate.mxfp4_file_storage = std::move(clustered_gate_storage);
    TensorData clustered_down;
    clustered_down.dtype = DType::MxFp4;
    clustered_down.shape = {2, 32};
    auto clustered_down_storage = std::make_shared<MxFp4FileStorage>();
    clustered_down_storage->blocks_path = clustered_path.string();
    clustered_down_storage->blocks_offset = 48;
    clustered_down_storage->blocks_size = 32;
    clustered_down_storage->scales_path = clustered_path.string();
    clustered_down_storage->scales_offset = 2;
    clustered_down_storage->scales_size = 2;
    clustered_down.mxfp4_file_storage = std::move(clustered_down_storage);
    ExpertCache clustered_cache(102, 1, {}, ExpertCacheBufferedReads);
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
    fragmented_gate_storage->blocks_size = 32;
    fragmented_gate_storage->scales_path = fragmented_path.string();
    fragmented_gate_storage->scales_offset = 33;
    fragmented_gate_storage->scales_size = 2;
    fragmented_gate_storage->secondary_blocks_path = fragmented_path.string();
    fragmented_gate_storage->secondary_blocks_offset = 36;
    fragmented_gate_storage->secondary_blocks_size = 32;
    fragmented_gate_storage->secondary_scales_path = fragmented_path.string();
    fragmented_gate_storage->secondary_scales_offset = 69;
    fragmented_gate_storage->secondary_scales_size = 2;
    fragmented_gate_storage->interleave_rows = true;
    fragmented_gate.mxfp4_file_storage = std::move(fragmented_gate_storage);
    TensorData fragmented_down;
    fragmented_down.dtype = DType::MxFp4;
    fragmented_down.shape = {2, 32};
    auto fragmented_down_storage = std::make_shared<MxFp4FileStorage>();
    fragmented_down_storage->blocks_path = fragmented_path.string();
    fragmented_down_storage->blocks_offset = 72;
    fragmented_down_storage->blocks_size = 32;
    fragmented_down_storage->scales_path = fragmented_path.string();
    fragmented_down_storage->scales_offset = 105;
    fragmented_down_storage->scales_size = 2;
    fragmented_down.mxfp4_file_storage = std::move(fragmented_down_storage);
    ExpertCache fragmented_cache(102, 1, {}, ExpertCacheBufferedReads);
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
    invalid_secondary_storage->secondary_blocks_size = 16;
    invalid_secondary.mxfp4_file_storage = std::move(invalid_secondary_storage);
    ExpertCache invalid_secondary_cache(50, 1, {}, ExpertCacheBufferedReads);
    auto invalid_secondary_pair = invalid_secondary_cache.acquire_pair(invalid_secondary, packed_down);
    check(!invalid_secondary_pair);
    check(invalid_secondary_pair.error().code == ErrorCode::InvalidModel);

    auto victim = std::make_shared<TestExpertVictimCache>(68);
    ExpertCache tiered(34, 1, victim, ExpertCacheMemoryMapRanges);
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

    if (get_gpu_count() > 0)
    {
        auto gpu_source = cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(gpu_source));
        auto gpu_victim = create_vulkan_victim_cache(
            68,
            automatic_vulkan_device_index,
            context_instance,
            g_test_optimization_flags);
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

    ExpertCache concurrent(68, 2);
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
        ExpertCache any_ready(68, 2, delayed);
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
        check(static_cast<bool>(leases[1].gate_up_operator != nullptr));
        check(static_cast<bool>(leases[1].down_operator != nullptr));

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
        check(static_cast<bool>(slow_lease[0].gate_up_operator != nullptr));
        check(static_cast<bool>(slow_lease[0].down_operator != nullptr));
        check(static_cast<bool>(any_ready.statistics().misses == 2));

        ExpertCache front_ready(68, 2, delayed);
        leases = {};
        acquired = front_ready.wait_acquire_ready_pairs(requests, leases, false);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value() == 2));
        check(static_cast<bool>(leases[0].gate_up));
        check(static_cast<bool>(leases[1].gate_up));
        check(static_cast<bool>(leases[0].gate_up_operator != nullptr));
        check(static_cast<bool>(leases[0].down_operator != nullptr));
        check(static_cast<bool>(leases[1].gate_up_operator != nullptr));
        check(static_cast<bool>(leases[1].down_operator != nullptr));
    }

    ExpertCache speculative(68, 1);
    check(static_cast<bool>(speculative.prefetch_pair(gate_zero, down_zero)));
    speculative.wait_for_background_work();
    check(speculative.is_ready(gate_zero, down_zero));
    auto prefetched = speculative.acquire_pair(gate_zero, down_zero);
    check(static_cast<bool>(prefetched));
    check(static_cast<bool>(speculative.statistics().speculative_reads == 1));
    check(static_cast<bool>(speculative.statistics().queued_reads == 1));

    ExpertCache adaptive_replacement(68, 1);
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
        check(static_cast<bool>(adapted.arc_recent_target_size == 34));
    }
    {
        auto frequent_ghost = adaptive_replacement.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(frequent_ghost));
        const ExpertCacheStatistics adapted = adaptive_replacement.statistics();
        check(static_cast<bool>(adapted.arc_frequent_ghost_hits == 1));
        check(static_cast<bool>(adapted.arc_recent_target_size == 0));
        check(static_cast<bool>(adapted.arc_recent_size + adapted.arc_frequent_size == adapted.resident_size));
    }

    ExpertCache pressure(34, 1);
    auto pinned = pressure.acquire_pair(gate_zero, down_zero);
    check(static_cast<bool>(pinned));
    check(static_cast<bool>(pressure.prefetch_pair(gate_one, down_one)));
    check(static_cast<bool>(pressure.statistics().speculative_reads == 0));
    check(static_cast<bool>(pressure.statistics().dropped_speculative_admissions == 1));
    auto exhausted = pressure.acquire_pair(gate_one, down_one);
    check(static_cast<bool>(!exhausted));
    check(static_cast<bool>(exhausted.error().code == ErrorCode::InvalidArgument));

    ExpertCache speculative_eviction(34, 1, {}, ExpertCacheAllowSpeculativeEviction);
    {
        auto resident = speculative_eviction.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(resident));
    }
    check(static_cast<bool>(speculative_eviction.prefetch_pair(gate_one, down_one)));
    speculative_eviction.wait_for_background_work();
    check(static_cast<bool>(speculative_eviction.is_ready(gate_one, down_one)));
    check(static_cast<bool>(speculative_eviction.statistics().speculative_reads == 1));
    check(static_cast<bool>(speculative_eviction.statistics().dropped_speculative_admissions == 0));

    ExpertCache variable_size(52, 1);
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
    check(static_cast<bool>(variable_statistics.resident_size == 52));
    check(static_cast<bool>(variable_statistics.arc_recent_size + variable_statistics.arc_frequent_size == variable_statistics.resident_size));

    ExpertCache oversized_prefetch(17, 1);
    check(static_cast<bool>(oversized_prefetch.prefetch_pair(small_gate, small_down)));
    check(static_cast<bool>(oversized_prefetch.statistics().dropped_speculative_admissions == 1));

    const TensorData truncated_gate = file_backed(60, 0);
    ExpertCache retryable(34, 1);
    auto failed_read = retryable.acquire_pair(truncated_gate, down_zero);
    check(static_cast<bool>(!failed_read));
    check(static_cast<bool>(failed_read.error().code == ErrorCode::IoError));
    auto retried_read = retryable.acquire_pair(truncated_gate, down_zero);
    check(static_cast<bool>(!retried_read));
    check(static_cast<bool>(retried_read.error().code == ErrorCode::IoError));
    check(static_cast<bool>(retryable.statistics().misses == 2));
    check(static_cast<bool>(retryable.statistics().resident_size == 0));

    ExpertCache undersized(33);
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
        storage->blocks_size = large_blocks_per_tensor;
        storage->scales_path = large_scales_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_size = large_scales_per_tensor;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    const TensorData large_gate = large_file_backed(0, 0);
    const TensorData large_down = large_file_backed(large_blocks_per_tensor, large_scales_per_tensor);
    const uint64_t large_pair_bytes = (large_blocks_per_tensor + large_scales_per_tensor) * 2;
    ExpertCache large_reads(large_pair_bytes, 2, {}, true);
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
    options.num_threads = 2;
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
         index < first_reference.value().logits.size();
         ++index)
    {
        check_near(
            prefill_results[0].value().logits[index],
            first_reference.value().logits[index],
            1e-5f);
    }
    for (size_t index = 0;
         index < second_reference.value().logits.size();
         ++index)
    {
        check_near(
            prefill_results[1].value().logits[index],
            second_reference.value().logits[index],
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
    for (size_t index = 0; index < reference_second.value().logits.size(); ++index)
    {
        check_near(ordered_second_result[0].value().logits[index], reference_second.value().logits[index], 1e-5f);
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
    check(static_cast<bool>(statistics.num_threads >= 1));
    check(static_cast<bool>(statistics.num_threads <= options.num_threads));
    check(static_cast<bool>(statistics.prefill_batches == 1));
    check(static_cast<bool>(statistics.staged_prefill_batches == 1));
    check(static_cast<bool>(statistics.decode_batches == 4));
    check(static_cast<bool>(statistics.staged_decode_batches == 1));

    check(static_cast<bool>(prefill_first.value()->reset()));
    check(static_cast<bool>(prefill_second.value()->reset()));
    check(static_cast<bool>(prefill_reference_first.value()->reset()));
    check(static_cast<bool>(prefill_reference_second.value()->reset()));
    auto staggered_third = runtime.create_session(model.value());
    auto staggered_reference_third = runtime.create_session(model.value());
    check(static_cast<bool>(staggered_third));
    check(static_cast<bool>(staggered_reference_third));

    auto staggered_scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(staggered_scheduler));
    const std::array<SessionPtr, 3> staggered_sessions = {
        prefill_first.value(),
        prefill_second.value(),
        staggered_third.value(),
    };
    const std::array<SessionPtr, 3> staggered_reference_sessions = {
        prefill_reference_first.value(),
        prefill_reference_second.value(),
        staggered_reference_third.value(),
    };
    const std::array<std::vector<int32_t>, 3> staggered_prompts = {
        std::vector<int32_t>{0},
        std::vector<int32_t>{1, 0, 1},
        std::vector<int32_t>{0, 1},
    };
    std::vector<PrefillBatchRequest> staggered_requests;
    staggered_requests.reserve(staggered_sessions.size());
    for (size_t index = 0; index < staggered_sessions.size(); ++index)
        staggered_requests.push_back({staggered_sessions[index], staggered_prompts[index]});
    const std::vector<Result<PrefillResult>> staggered_results = staggered_scheduler.value()->submit_prefill(std::move(staggered_requests)).get();

    check(staggered_results.size() == staggered_sessions.size());
    for (size_t index = 0; index < staggered_sessions.size(); ++index)
    {
        auto expected = staggered_reference_sessions[index]->prefill(staggered_prompts[index]);
        check(static_cast<bool>(expected));
        check(static_cast<bool>(staggered_results[index]));
        check(staggered_results[index].value().processed_tokens == staggered_prompts[index].size());
        check(staggered_sessions[index]->sequence_length() == staggered_prompts[index].size());
        check(staggered_results[index].value().logits.size() == expected.value().logits.size());
        for (size_t logit = 0; logit < expected.value().logits.size(); ++logit)
            check_near(staggered_results[index].value().logits[logit], expected.value().logits[logit], 1e-5f);
    }

    auto different_input_first = runtime.create_session(model.value());
    auto different_input_second = runtime.create_session(model.value());
    check(static_cast<bool>(different_input_first));
    check(static_cast<bool>(different_input_second));
    const SchedulerStatistics before_different_inputs = scheduler.value()->statistics();
    auto different_input_future = scheduler.value()->submit_decode({
        {different_input_first.value(), 0},
        {different_input_second.value(), 1},
    });
    const std::vector<Result<DecodeResult>> different_input_results = different_input_future.get();
    check(static_cast<bool>(different_input_results.size() == 2));
    check(static_cast<bool>(different_input_results[0]));
    check(static_cast<bool>(different_input_results[1]));
    const SchedulerStatistics after_different_inputs = scheduler.value()->statistics();
    check(static_cast<bool>(
        after_different_inputs.staged_decode_batches
        == before_different_inputs.staged_decode_batches + 1));

    SchedulerOptions large_scheduler_options;
    large_scheduler_options.num_threads = 1025;
    auto large_scheduler = runtime.create_scheduler(large_scheduler_options);
    check(static_cast<bool>(large_scheduler));
    if (large_scheduler)
    {
        const SchedulerStatistics statistics = large_scheduler.value()->statistics();
        check(static_cast<bool>(statistics.num_threads >= 1));
        check(static_cast<bool>(statistics.num_threads <= large_scheduler_options.num_threads));
    }

    if (has_flag(runtime.info().flags, RuntimeVulkanCpu))
    {
        AttentionPackage attention_package;
        Option hybrid_options;
        hybrid_options.hybrid_mode = HybridMode::HybridExperts;
        Option cpu_options;
        cpu_options.hybrid_mode = HybridMode::CpuOnly;
        auto hybrid_model = runtime.load_model(attention_package.path(), hybrid_options);
        auto cpu_model = runtime.load_model(attention_package.path(), cpu_options);
        check(static_cast<bool>(hybrid_model));
        check(static_cast<bool>(cpu_model));

        SchedulerOptions pipeline_options;
        pipeline_options.num_threads = 4;
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
                check(static_cast<bool>(pipeline_results[session_index].value().logits.size() == cpu_result.value().logits.size()));
                for (size_t logit = 0; logit < cpu_result.value().logits.size(); ++logit)
                {
                    check_near(pipeline_results[session_index].value().logits[logit], cpu_result.value().logits[logit], 1e-4f);
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
        check(static_cast<bool>(pipeline_statistics.decode_batches == pipeline_rounds));
        check(static_cast<bool>(pipeline_statistics.staged_decode_batches == pipeline_rounds));
    }
}

void test_batch_scheduler_submission_contract()
{
    AttentionPackage package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt);
    auto other_model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));
    check(static_cast<bool>(other_model));
    auto first = runtime.create_session(model.value());
    auto second = runtime.create_session(model.value());
    auto other = runtime.create_session(other_model.value());
    check(static_cast<bool>(first));
    check(static_cast<bool>(second));
    check(static_cast<bool>(other));
    check(static_cast<bool>(first.value()->prefill(std::vector<int32_t>{0})));
    check(static_cast<bool>(second.value()->prefill(std::vector<int32_t>{1})));
    const SessionStatistics first_before = first.value()->statistics();
    const SessionStatistics second_before = second.value()->statistics();

    SchedulerOptions options;
    options.num_threads = 2;
    auto validation_scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(validation_scheduler));

    // Empty submissions are no-ops; malformed batches must not enqueue a valid prefix.
    check(validation_scheduler.value()->submit_prefill({}).get().empty());
    check(validation_scheduler.value()->submit_decode({}).get().empty());
    const auto reject_prefill = [&](std::vector<PrefillBatchRequest> requests) {
        const size_t count = requests.size();
        const std::vector<Result<PrefillResult>> results = validation_scheduler.value()->submit_prefill(std::move(requests)).get();
        check(results.size() == count);
        for (const Result<PrefillResult>& result : results)
        {
            check(!result);
            check(result.error().code == ErrorCode::InvalidArgument);
        }
    };
    const auto reject_decode = [&](std::vector<DecodeBatchRequest> requests) {
        const size_t count = requests.size();
        const std::vector<Result<DecodeResult>> results = validation_scheduler.value()->submit_decode(std::move(requests)).get();
        check(results.size() == count);
        for (const Result<DecodeResult>& result : results)
        {
            check(!result);
            check(result.error().code == ErrorCode::InvalidArgument);
        }
    };
    reject_prefill({{first.value(), {1}}, {nullptr, {0}}});
    reject_prefill({{first.value(), {1}}, {second.value(), {}}});
    reject_prefill({{first.value(), {1}}, {first.value(), {0}}});
    reject_prefill({{first.value(), {1}}, {other.value(), {0}}});
    reject_decode({{first.value(), 1}, {nullptr, 0}});
    reject_decode({{first.value(), 1}, {first.value(), 0}});
    reject_decode({{first.value(), 1}, {second.value(), 2}});
    validation_scheduler.value().reset();

    const auto check_unchanged = [](const SessionPtr& session, const SessionStatistics& before) {
        const SessionStatistics after = session->statistics();
        check(session->sequence_length() == 1);
        check(after.prefill_tokens == before.prefill_tokens);
        check(after.decode_tokens == before.decode_tokens);
        check(after.expert_assignments == before.expert_assignments);
        check(after.expert_token_counts == before.expert_token_counts);
        check(after.kv_cache_logical_size == before.kv_cache_logical_size);
        check(after.kv_cache_allocated_size == before.kv_cache_allocated_size);
    };
    check_unchanged(first.value(), first_before);
    check_unchanged(second.value(), second_before);
    check(other.value()->sequence_length() == 0);
    check(other.value()->statistics().prefill_tokens == 0);
    check(other.value()->statistics().decode_tokens == 0);

    auto ordered = runtime.create_session(model.value());
    auto reference = runtime.create_session(model.value());
    check(static_cast<bool>(ordered));
    check(static_cast<bool>(reference));
    auto draining_scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(draining_scheduler));
    const std::vector<int32_t> prompt = {0, 1};
    auto queued_prefill = draining_scheduler.value()->submit_prefill({{ordered.value(), prompt}});
    constexpr std::array<int32_t, 6> tokens = {1, 0, 1, 1, 0, 0};
    std::vector<std::future<std::vector<Result<DecodeResult>>>> queued_decodes;
    queued_decodes.reserve(tokens.size());
    for (int32_t token : tokens)
        queued_decodes.push_back(draining_scheduler.value()->submit_decode({{ordered.value(), token}}));

    // Destruction must complete all accepted work without requiring future.get().
    draining_scheduler.value().reset();
    check(queued_prefill.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
    for (std::future<std::vector<Result<DecodeResult>>>& future : queued_decodes)
        check(future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
    const std::vector<Result<PrefillResult>> prefilled = queued_prefill.get();
    check(prefilled.size() == 1);
    check(static_cast<bool>(prefilled[0]));
    check(prefilled[0].value().processed_tokens == prompt.size());
    auto reference_prefill = reference.value()->prefill(prompt);
    check(static_cast<bool>(reference_prefill));
    check(prefilled[0].value().logits.size() == reference_prefill.value().logits.size());
    for (size_t logit = 0; logit < reference_prefill.value().logits.size(); ++logit)
        check_near(prefilled[0].value().logits[logit], reference_prefill.value().logits[logit], 1e-5f);
    for (size_t index = 0; index < tokens.size(); ++index)
    {
        const std::vector<Result<DecodeResult>> results = queued_decodes[index].get();
        auto expected = reference.value()->decode(tokens[index]);
        check(results.size() == 1);
        check(static_cast<bool>(results[0]));
        check(static_cast<bool>(expected));
        check(results[0].value().sequence_length == prompt.size() + index + 1);
        check(results[0].value().logits.size() == expected.value().logits.size());
        for (size_t logit = 0; logit < expected.value().logits.size(); ++logit)
            check_near(results[0].value().logits[logit], expected.value().logits[logit], 1e-5f);
    }
    check(ordered.value()->sequence_length() == prompt.size() + tokens.size());
    check(ordered.value()->statistics().prefill_tokens == prompt.size());
    check(ordered.value()->statistics().decode_tokens == tokens.size());

    auto staged_first = runtime.create_session(model.value());
    auto staged_second = runtime.create_session(model.value());
    auto reference_first = runtime.create_session(model.value());
    auto reference_second = runtime.create_session(model.value());
    check(static_cast<bool>(staged_first));
    check(static_cast<bool>(staged_second));
    check(static_cast<bool>(reference_first));
    check(static_cast<bool>(reference_second));
    auto resubmit_scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(resubmit_scheduler));
    const std::vector<int32_t> first_prompt = {0, 1};
    const std::vector<int32_t> second_prompt = {1};
    const std::vector<PrefillBatchRequest> requests = {
        {staged_first.value(), first_prompt},
        {staged_second.value(), second_prompt},
    };
    constexpr uint32_t rounds = 4;
    auto pending = resubmit_scheduler.value()->submit_prefill(requests);
    for (uint32_t round = 0; round < rounds; ++round)
    {
        const std::vector<Result<PrefillResult>> results = pending.get();
        // A ready future must mean the staged Session reservations are already released.
        if (round + 1 < rounds)
            pending = resubmit_scheduler.value()->submit_prefill(requests);
        check(results.size() == 2);
        check(static_cast<bool>(results[0]));
        check(static_cast<bool>(results[1]));
        check(results[0].value().processed_tokens == first_prompt.size());
        check(results[1].value().processed_tokens == second_prompt.size());
        auto expected_first = reference_first.value()->prefill(first_prompt);
        auto expected_second = reference_second.value()->prefill(second_prompt);
        check(static_cast<bool>(expected_first));
        check(static_cast<bool>(expected_second));
        check(results[0].value().logits.size() == expected_first.value().logits.size());
        check(results[1].value().logits.size() == expected_second.value().logits.size());
        for (size_t logit = 0; logit < expected_first.value().logits.size(); ++logit)
            check_near(results[0].value().logits[logit], expected_first.value().logits[logit], 1e-5f);
        for (size_t logit = 0; logit < expected_second.value().logits.size(); ++logit)
            check_near(results[1].value().logits[logit], expected_second.value().logits[logit], 1e-5f);
    }
    check(staged_first.value()->sequence_length() == rounds * first_prompt.size());
    check(staged_second.value()->sequence_length() == rounds * second_prompt.size());
    check(staged_first.value()->statistics().prefill_tokens == rounds * first_prompt.size());
    check(staged_second.value()->statistics().prefill_tokens == rounds * second_prompt.size());
    check(resubmit_scheduler.value()->statistics().staged_prefill_batches == rounds);
}

void test_independent_batch_scheduler()
{
    AttentionPackage package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));
    std::array<SessionPtr, 2> sessions;
    std::array<SessionPtr, 2> references;
    for (size_t i = 0; i < sessions.size(); i++)
    {
        auto session = runtime.create_session(model.value());
        auto reference = runtime.create_session(model.value());
        check(static_cast<bool>(session));
        check(static_cast<bool>(reference));
        sessions[i] = session.value();
        references[i] = reference.value();
    }

    SchedulerOptions options;
    options.num_threads = 2;
    options.use_staged_decode = false;
    auto scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(scheduler));
    constexpr size_t rounds = 4;
    std::vector<std::future<std::vector<Result<DecodeResult>>>> pending;
    for (size_t round = 0; round < rounds; round++)
    {
        std::vector<DecodeBatchRequest> requests;
        for (size_t i = 0; i < sessions.size(); i++)
            requests.push_back({sessions[(round + i) % sessions.size()], static_cast<int32_t>(i)});
        pending.push_back(scheduler.value()->submit_decode(std::move(requests)));
    }
    auto mixed = scheduler.value()->submit_decode({{sessions[0], 2}, {sessions[1], 0}});
    check(scheduler.value()->statistics().decode_batches == rounds + 1);
    check(scheduler.value()->statistics().staged_decode_batches == 0);
    scheduler.value().reset();

    // Request order and per-Session FIFO must hold even when batch order alternates.
    for (size_t round = 0; round < rounds; round++)
    {
        check(pending[round].wait_for(std::chrono::seconds(0)) == std::future_status::ready);
        const auto results = pending[round].get();
        check(results.size() == sessions.size());
        for (size_t i = 0; i < sessions.size(); i++)
        {
            auto expected = references[(round + i) % sessions.size()]->decode(static_cast<int32_t>(i));
            check(static_cast<bool>(expected));
            check(static_cast<bool>(results[i]));
            check(results[i].value().sequence_length == round + 1);
            check(results[i].value().logits.size() == expected.value().logits.size());
            for (size_t j = 0; j < expected.value().logits.size(); j++)
                check_near(results[i].value().logits[j], expected.value().logits[j], 1e-5f);
        }
    }
    const auto mixed_results = mixed.get();
    check(mixed_results.size() == 2);
    check(!mixed_results[0]);
    check(mixed_results[0].error().code == ErrorCode::InvalidArgument);
    check(static_cast<bool>(mixed_results[1]));
    check(mixed_results[1].value().sequence_length == rounds + 1);
    check(sessions[0]->sequence_length() == rounds);
    check(sessions[1]->sequence_length() == rounds + 1);
}

void test_staged_bfloat16_dispatch_telemetry()
{
    Bfloat16StagedBatchPackage package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    opt.optimization_flags &= ~OptimizationNcnnCpuBfloat16Linear;
    SchedulerOptions scheduler_options;
    scheduler_options.num_threads = 2;
    auto scheduler = runtime.create_scheduler(scheduler_options);
    check(static_cast<bool>(scheduler));

    auto run_batch = [&](bool batched_enabled) {
        Option batch_options = opt;
        const uint64_t batch_flags = OptimizationCpuBfloat16Batched;
        if (batched_enabled)
            batch_options.optimization_flags |= batch_flags;
        else
            batch_options.optimization_flags &= ~batch_flags;
        auto batch_model = runtime.load_model(package.path(), batch_options);
        check(static_cast<bool>(batch_model));
        check(model_compiled(*batch_model.value()).opt.optimization_flags
              == batch_options.optimization_flags);
        std::vector<SessionPtr> sessions;
        std::vector<DecodeBatchRequest> requests;
        sessions.reserve(4);
        requests.reserve(4);
        for (uint32_t index = 0; index < 4; ++index)
        {
            auto session = runtime.create_session(batch_model.value());
            check(static_cast<bool>(session));
            sessions.push_back(session.value());
            requests.push_back({session.value(), static_cast<int32_t>(index)});
        }
        const std::vector<Result<DecodeResult>> results = scheduler.value()->submit_decode(std::move(requests)).get();
        check(static_cast<bool>(results.size() == sessions.size()));
        for (const Result<DecodeResult>& result : results)
            check(static_cast<bool>(result));
        return sessions;
    };

    const std::vector<SessionPtr> enabled_sessions = run_batch(true);
    const bool kernel_available = std::string(bfloat16_batched_linear_kernel_name(
                                      opt.optimization_flags
                                      | OptimizationCpuBfloat16Batched))
                                  != "unavailable";
    for (const SessionPtr& session : enabled_sessions)
    {
        const uint64_t dispatches = session->statistics().cpu_bfloat16_batched_linear_dispatches;
        check(kernel_available ? dispatches == 2 : dispatches == 0);
    }

    const std::vector<SessionPtr> disabled_sessions = run_batch(false);
    for (const SessionPtr& session : disabled_sessions)
    {
        check(static_cast<bool>(
            session->statistics().cpu_bfloat16_batched_linear_dispatches
            == 0));
    }

    if (kernel_available)
    {
        std::vector<SessionPtr> first_sessions;
        std::vector<SessionPtr> second_sessions;
        std::vector<DecodeBatchRequest> first_requests;
        std::vector<DecodeBatchRequest> second_requests;
        Option enabled_options = opt;
        enabled_options.optimization_flags |= OptimizationCpuBfloat16Batched;
        auto enabled_model = runtime.load_model(package.path(), enabled_options);
        check(static_cast<bool>(enabled_model));
        check(model_compiled(*enabled_model.value()).opt.optimization_flags
              == enabled_options.optimization_flags);
        for (uint32_t index = 0; index < 4; ++index)
        {
            auto first = runtime.create_session(enabled_model.value());
            auto second = runtime.create_session(enabled_model.value());
            check(static_cast<bool>(first));
            check(static_cast<bool>(second));
            first_sessions.push_back(first.value());
            second_sessions.push_back(second.value());
            first_requests.push_back(
                {first.value(), static_cast<int32_t>(index)});
            second_requests.push_back(
                {second.value(), static_cast<int32_t>(3 - index)});
        }
        auto first_future = scheduler.value()->submit_decode(std::move(first_requests));
        auto second_future = scheduler.value()->submit_decode(std::move(second_requests));
        const std::vector<Result<DecodeResult>> first_results = first_future.get();
        const std::vector<Result<DecodeResult>> second_results = second_future.get();
        check(static_cast<bool>(first_results.size() == 4));
        check(static_cast<bool>(second_results.size() == 4));
        for (const Result<DecodeResult>& result : first_results)
            check(static_cast<bool>(result));
        for (const Result<DecodeResult>& result : second_results)
            check(static_cast<bool>(result));
        for (const SessionPtr& session : first_sessions)
        {
            check(static_cast<bool>(
                session->statistics().cpu_bfloat16_batched_linear_dispatches
                == 2));
        }
        for (const SessionPtr& session : second_sessions)
        {
            check(static_cast<bool>(
                session->statistics().cpu_bfloat16_batched_linear_dispatches
                == 2));
        }
    }
}

void test_chunked_prefill_statistics_commit_on_success()
{
    Bfloat16StagedBatchPackage package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    opt.optimization_flags &= ~OptimizationNcnnCpuBfloat16Linear;
    opt.optimization_flags |= OptimizationCpuBfloat16Batched;
    auto model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));

    SessionOptions session_options;
    session_options.prefill_chunk_size = 4;
    auto control = runtime.create_session(model.value(), session_options);
    auto session = runtime.create_session(model.value(), session_options);
    check(static_cast<bool>(control));
    check(static_cast<bool>(session));

    const std::array<int32_t, 4> prompt = {0, 1, 2, 3};
    const bool kernel_available = std::string(bfloat16_batched_linear_kernel_name(opt.optimization_flags)) != "unavailable";
    for (const SessionPtr& current : {control.value(), session.value()})
    {
        auto prefilled = current->prefill(prompt);
        check(static_cast<bool>(prefilled));
        check(prefilled.value().processed_tokens == prompt.size());
        check(current->sequence_length() == prompt.size());
        const SessionStatistics statistics = current->statistics();
        check(statistics.prefill_tokens == prompt.size());
        check(statistics.expert_assignments == prompt.size());
        check(statistics.expert_batches == 1);
        check(kernel_available ? statistics.cpu_bfloat16_batched_linear_dispatches == 2
                               : statistics.cpu_bfloat16_batched_linear_dispatches == 0);
    }

    const SessionStatistics before = session.value()->statistics();
    const uint64_t sequence_length_before = session.value()->sequence_length();
    // The first chunk completes before the invalid token is checked in the next chunk.
    const std::array<int32_t, 5> invalid_prompt = {0, 1, 2, 3, 4};
    auto failed = session.value()->prefill(invalid_prompt);
    check(!failed);
    check(failed.error().code == ErrorCode::InvalidArgument);
    check(failed.error().message == "token id is outside the model vocabulary");
    const SessionStatistics after = session.value()->statistics();
    check(session.value()->sequence_length() == sequence_length_before);
    check(after.prefill_tokens == before.prefill_tokens);
    check(after.decode_tokens == before.decode_tokens);
    check(after.expert_assignments == before.expert_assignments);
    check(after.expert_batches == before.expert_batches);
    check(after.expert_token_counts == before.expert_token_counts);
    check(after.cpu_bfloat16_batched_linear_dispatches == before.cpu_bfloat16_batched_linear_dispatches);

    // Published statistics are transactional; this does not assert execution-state rollback.
    check(static_cast<bool>(session.value()->reset()));
    check(session.value()->sequence_length() == 0);
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
    check(static_cast<bool>(chunked.value().logits.size() == single_batch.value().logits.size()));
    for (size_t index = 0; index < single_batch.value().logits.size(); ++index)
    {
        check_near(chunked.value().logits[index], single_batch.value().logits[index], 1e-5f);
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

    check_near(result.value().logits[0], hidden_x / final_scale, 1e-5f);
    check_near(result.value().logits[1], hidden_y / final_scale, 1e-5f);
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 2));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 2));
    check(static_cast<bool>(session.value()->statistics().expert_token_counts == std::vector<uint64_t>({1, 1, 0})));
    if (has_flag(runtime.info().flags, RuntimeOpenmp))
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
    check_near(result.value().logits[0], expected, 1e-5f);
    check_near(result.value().logits[1], 0.0f, 1e-5f);
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
    check(static_cast<bool>(model.value()->descriptor().layers[0].attention.kind == AttentionKind::Standard));

    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));
    const std::vector<int32_t> prompt = {0};
    auto prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill));
    auto cached_decode = session.value()->decode(1);
    check(static_cast<bool>(cached_decode));
    check(static_cast<bool>(cached_decode.value().logits[0] > 0.1f));

    check(static_cast<bool>(session.value()->reset()));
    auto uncached_decode = session.value()->decode(1);
    check(static_cast<bool>(uncached_decode));
    check_near(uncached_decode.value().logits[0], 0.0f, 1e-6f);
}

void test_bfloat16_ring_kv_cache()
{
    AttentionPackage float32_package;
    AttentionPackage bfloat16_package(true);
    TestRuntime runtime;
    Option cpu_options;
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
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_logical_size == float32_session.value()->statistics().kv_cache_logical_size / 2));
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_allocated_size == float32_session.value()->statistics().kv_cache_allocated_size / 2));

    for (uint32_t index = 0; index < 16; ++index)
        check(static_cast<bool>(bfloat16_session.value()->decode(static_cast<int32_t>(index % 2))));
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_allocated_size <= bfloat16_session.value()->statistics().kv_cache_logical_size * 16));
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
    auto graph_driven_model = compile_model(
        std::move(descriptor).value(),
        std::move(mapping).value(), HybridMode::CpuOnly);
    if (!graph_driven_model)
    {
        throw std::runtime_error("graph-driven model compilation failed: " + graph_driven_model.error().message);
    }
    check(graph_driven_model.value().graph.layer_plans[0].attention.kind
          == AttentionKind::Standard);
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
#if NCNN_MOE_WITH_NCNN
    const AttentionBlockPlan& attention = compiled.graph.layer_plans.front().attention;
    const CompiledOperator& query_operator = compiled.operators.at_weight(attention.query_weight);
    check(static_cast<bool>(query_operator.linear));
    check(!query_operator.linear->uses_vulkan());
#endif
    check(!support_vulkan_attention(compiled.operators, compiled.graph.layer_plans.front().attention));

    // Hybrid models can keep QSA or gated-residual attention on the CPU.
    CompiledModel hybrid_compiled = compiled;
    hybrid_compiled.opt.hybrid_mode = HybridMode::HybridExperts;
    hybrid_compiled.opt.optimization_flags |= OptimizationVulkanAttention;
    auto rebuilt_graph = build_graph(hybrid_compiled, false);
    check(static_cast<bool>(rebuilt_graph));
    const ExecutionNode* attention_node = nullptr;
    for (const ExecutionNode& node : hybrid_compiled.graph.nodes)
    {
        if (node.type == ExecutionNodeType::Attention)
        {
            attention_node = &node;
            break;
        }
    }
    check(attention_node != nullptr);
    check(attention_node->backend == ExecutionBackend::Cpu);
    check(attention_node->backend_mask == ExecutionBackendCpu);
    auto full_attention = execute_attention_block_into(
        compiled.weights,
        compiled.operators,
        compiled.graph.layer_plans.front().attention,
        ExecutionBackend::Cpu,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        7,
        full_cache,
        full_scratch,
        context_hidden,
        full_output,
        g_test_optimization_flags);
    check(static_cast<bool>(full_attention));
    auto appended_context = append_attention_context_into(
        compiled.weights,
        compiled.operators,
        compiled.graph.layer_plans.front().attention,
        ExecutionBackend::Cpu,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        7,
        context_cache,
        context_scratch,
        context_hidden,
        g_test_optimization_flags);
    check(static_cast<bool>(appended_context));
    check(full_cache.start_position == context_cache.start_position);
    check(full_cache.token_count == context_cache.token_count);
    check(full_cache.first_slot == context_cache.first_slot);
    check(full_cache.keys == context_cache.keys);
    check(full_cache.values == context_cache.values);

    CpuBatch long_hidden(80, 2);
    for (size_t row = 0; row < long_hidden.rows(); ++row)
    {
        long_hidden.row(row)[0] = static_cast<float>(static_cast<int>(row % 13) - 6) * 0.0625f;
        long_hidden.row(row)[1] = static_cast<float>(static_cast<int>((row * 3) % 17) - 8) * 0.03125f;
    }
    const uint64_t attention_reference_flags = g_test_optimization_flags
                                               & ~OptimizationCpuFlashAttention
                                               & ~OptimizationCpuSplitKvAttention;
    CpuLayerCache reference_long_cache;
    CpuLayerCache flash_long_cache;
    CpuAttentionExecutionScratch reference_long_scratch;
    CpuAttentionExecutionScratch flash_long_scratch;
    CpuBatch reference_long_output;
    CpuBatch flash_long_output;
    auto reference_long = execute_attention_block_into(
        compiled.weights,
        compiled.operators,
        compiled.graph.layer_plans.front().attention,
        ExecutionBackend::Cpu,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        0,
        reference_long_cache,
        reference_long_scratch,
        long_hidden,
        reference_long_output,
        attention_reference_flags);
    auto flash_long = execute_attention_block_into(
        compiled.weights,
        compiled.operators,
        compiled.graph.layer_plans.front().attention,
        ExecutionBackend::Cpu,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        0,
        flash_long_cache,
        flash_long_scratch,
        long_hidden,
        flash_long_output,
        g_test_optimization_flags);
    check(static_cast<bool>(reference_long));
    check(static_cast<bool>(flash_long));
    check(static_cast<bool>(reference_long_output.rows() == flash_long_output.rows()));
    check(static_cast<bool>(reference_long_output.columns() == flash_long_output.columns()));
    for (size_t row = 0; row < reference_long_output.rows(); ++row)
        for (uint32_t column = 0; column < reference_long_output.columns(); ++column)
            check_near(flash_long_output.row(row)[column], reference_long_output.row(row)[column], 1e-4f);

    CpuBatch split_prefix(512, 2);
    for (size_t row = 0; row < split_prefix.rows(); ++row)
    {
        split_prefix.row(row)[0] = static_cast<float>(static_cast<int>(row % 19) - 9) * 0.015625f;
        split_prefix.row(row)[1] = static_cast<float>(static_cast<int>((row * 5) % 23) - 11) * 0.015625f;
    }
    CpuLayerCache reference_split_cache;
    CpuLayerCache split_kv_cache;
    CpuAttentionExecutionScratch reference_split_scratch;
    CpuAttentionExecutionScratch split_kv_scratch;
    auto reference_context = append_attention_context_into(
        compiled.weights,
        compiled.operators,
        compiled.graph.layer_plans.front().attention,
        ExecutionBackend::Cpu,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        0,
        reference_split_cache,
        reference_split_scratch,
        split_prefix,
        attention_reference_flags);
    auto split_context = append_attention_context_into(
        compiled.weights,
        compiled.operators,
        compiled.graph.layer_plans.front().attention,
        ExecutionBackend::Cpu,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        0,
        split_kv_cache,
        split_kv_scratch,
        split_prefix,
        g_test_optimization_flags);
    check(static_cast<bool>(reference_context));
    check(static_cast<bool>(split_context));
    CpuBatch split_token(1, 2);
    split_token.row(0)[0] = 0.125f;
    split_token.row(0)[1] = -0.0625f;
    CpuBatch reference_split_output;
    CpuBatch split_kv_output;
    auto reference_decode = execute_attention_block_into(
        compiled.weights,
        compiled.operators,
        compiled.graph.layer_plans.front().attention,
        ExecutionBackend::Cpu,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        512,
        reference_split_cache,
        reference_split_scratch,
        split_token,
        reference_split_output,
        attention_reference_flags);
    auto split_decode = execute_attention_block_into(
        compiled.weights,
        compiled.operators,
        compiled.graph.layer_plans.front().attention,
        ExecutionBackend::Cpu,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        512,
        split_kv_cache,
        split_kv_scratch,
        split_token,
        split_kv_output,
        g_test_optimization_flags);
    check(static_cast<bool>(reference_decode));
    check(static_cast<bool>(split_decode));
    for (uint32_t column = 0; column < reference_split_output.columns(); ++column)
        check_near(split_kv_output.row(0)[column], reference_split_output.row(0)[column], 1e-4f);

    AttentionBlockPlan sliding_plan = compiled.graph.layer_plans.front().attention;
    sliding_plan.sliding_window = 2;
    std::array<CpuLayerCache, 1> sliding_context_cache;
    check(static_cast<bool>(begin_state_cache_transaction(
        sliding_context_cache,
        context_hidden.rows())));
    CpuAttentionExecutionScratch sliding_context_scratch;
    auto rejected_sliding_context = append_attention_context_into(
        compiled.weights,
        compiled.operators,
        sliding_plan,
        ExecutionBackend::Cpu,
        compiled.descriptor.norm_epsilon,
        compiled.descriptor.kv_cache_dtype,
        7,
        sliding_context_cache.front(),
        sliding_context_scratch,
        context_hidden,
        g_test_optimization_flags);
    check(static_cast<bool>(!rejected_sliding_context));
    check(static_cast<bool>(
        rejected_sliding_context.error().code
        == ErrorCode::UnsupportedModel));
    check(sliding_context_cache.front().token_count == 0);
    check(static_cast<bool>(finish_state_cache_transaction(
        sliding_context_cache,
        0)));

    TestRuntime runtime;
    Option cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto cpu_model = runtime.load_model(package.path(), cpu_options);
    check(static_cast<bool>(cpu_model));
    const CompiledModel& cpu_compiled = model_compiled(*cpu_model.value());
    const LayerDescriptor& layer = cpu_model.value()->descriptor().layers[0];
    check(static_cast<bool>(!has_flag(layer.attention.flags, AttentionDescriptorBias)));
    check(static_cast<bool>(!has_flag(layer.attention.flags, AttentionDescriptorSinks)));
    for (const ExecutionNode& node : cpu_compiled.graph.nodes)
        check(static_cast<bool>(node.backend == ExecutionBackend::Cpu));

    auto cpu_session = runtime.create_session(cpu_model.value());
    check(static_cast<bool>(cpu_session));
    const std::vector<int32_t> prompt = {0, 1, 0};
    auto cpu_prefill = cpu_session.value()->prefill(prompt);
    check(static_cast<bool>(cpu_prefill));

    if (has_flag(runtime.info().flags, RuntimeVulkanAttention))
    {
        Option hybrid_options;
        hybrid_options.hybrid_mode = HybridMode::HybridExperts;
        auto hybrid_model = runtime.load_model(package.path(), hybrid_options);
        check(static_cast<bool>(hybrid_model));
        const ExecutionGraph& hybrid_graph = model_compiled(*hybrid_model.value()).graph;
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
        check(static_cast<bool>(hybrid_prefill.value().logits.size() == cpu_prefill.value().logits.size()));
        for (size_t index = 0; index < cpu_prefill.value().logits.size(); ++index)
        {
            check_near(hybrid_prefill.value().logits[index], cpu_prefill.value().logits[index], 1e-4f);
        }
    }
}

void test_shared_expert_descriptor()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));
    const CompiledModel& compiled = model_compiled(*model.value());
    check(compiled.descriptor.layers.front().ffn.moe.shared_expert_count == 0);
    check(!compiled.graph.layer_plans.front().moe.has_shared_expert);
    MoeModelDescriptor baseline_descriptor = compiled.descriptor;
    check(static_cast<bool>(validate_model_descriptor(baseline_descriptor)));

    MoeModelDescriptor shared_descriptor = baseline_descriptor;
    shared_descriptor.layers.front().ffn.moe.shared_expert_count = 1;
    shared_descriptor.layers.front().ffn.moe.shared_expert_weight_dtype = DType::Float32;
    check(static_cast<bool>(validate_model_descriptor(shared_descriptor)));

    FixtureModelAdapter adapter;
    ModelPackage fixture;
    fixture.root = package.path();
    auto mapping = adapter.map_weights(fixture, compiled.descriptor);
    check(static_cast<bool>(mapping));
    for (const std::string& projection : {"gate", "up", "down"})
    {
        TensorData weight;
        weight.dtype = DType::Float32;
        weight.shape = projection == "down"
                           ? std::vector<uint32_t>{shared_descriptor.hidden_size, shared_descriptor.intermediate_size}
                           : std::vector<uint32_t>{shared_descriptor.intermediate_size, shared_descriptor.hidden_size};
        weight.float32_data.resize(static_cast<size_t>(shared_descriptor.hidden_size) * shared_descriptor.intermediate_size, 0.0f);
        mapping.value().emplace("layers.0.shared_expert." + projection + ".weight", std::move(weight));
    }
    auto shared_model = compile_model(shared_descriptor, std::move(mapping).value());
    check(static_cast<bool>(shared_model));
    const MoeBlockPlan& shared_plan = shared_model.value().graph.layer_plans.front().moe;
    check(shared_plan.has_shared_expert);
    check(shared_plan.shared_expert.gate_weight != invalid_tensor_handle);
    check(shared_plan.shared_expert.up_weight != invalid_tensor_handle);
    check(shared_plan.shared_expert.down_weight != invalid_tensor_handle);

    auto missing_shared_mapping = adapter.map_weights(fixture, compiled.descriptor);
    check(static_cast<bool>(missing_shared_mapping));
    auto missing_shared_model = compile_model(
        shared_descriptor,
        std::move(missing_shared_mapping).value());
    check(!missing_shared_model);
    check(missing_shared_model.error().code == ErrorCode::InvalidModel);
    check(missing_shared_model.error().message
          == "missing tensor: layers.0.shared_expert.gate.weight");

    MoeModelDescriptor gated_descriptor = baseline_descriptor;
    gated_descriptor.layers.front().ffn.moe.flags |= MoeDescriptorSharedExpertGate;
    auto gated_mapping = adapter.map_weights(fixture, compiled.descriptor);
    check(static_cast<bool>(gated_mapping));
    auto gated_model = compile_model(gated_descriptor, std::move(gated_mapping).value());
    check(!gated_model);
    check(gated_model.error().code == ErrorCode::InvalidModel);
    check(gated_model.error().message == "shared Expert gate requires a shared Expert");

    shared_descriptor.layers.front().ffn.moe.shared_expert_count = 2;
    check(static_cast<bool>(validate_model_descriptor(shared_descriptor)));
    auto multiple_mapping = adapter.map_weights(fixture, compiled.descriptor);
    check(static_cast<bool>(multiple_mapping));
    auto multiple_model = compile_model(shared_descriptor, std::move(multiple_mapping).value());
    check(!multiple_model);
    check(multiple_model.error().code == ErrorCode::UnsupportedModel);
    check(multiple_model.error().message == "the CPU runtime supports one shared Expert per MoE block");
}

void test_execution_graph_and_scheduler()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    Option options;
    options.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), options);
    check(static_cast<bool>(model));
    const CompiledModel& compiled = model_compiled(*model.value());
    check(static_cast<bool>(compiled.descriptor.model_type == "test_moe"));
    MoeModelDescriptor baseline_descriptor = compiled.descriptor;
    check(static_cast<bool>(validate_model_descriptor(baseline_descriptor)));
    MoeModelDescriptor missing_layers_descriptor = baseline_descriptor;
    missing_layers_descriptor.layers.clear();
    auto missing_layers_status = validate_model_descriptor(missing_layers_descriptor);
    check(!missing_layers_status);
    check(missing_layers_status.error().code == ErrorCode::InvalidModel);
    check(missing_layers_status.error().message == "model descriptor requires at least one layer");

    MoeModelDescriptor overflowing_speculative_ids_descriptor = baseline_descriptor;
    overflowing_speculative_ids_descriptor.speculative_layer_count = std::numeric_limits<uint32_t>::max();
    auto overflowing_speculative_ids_status = validate_model_descriptor(overflowing_speculative_ids_descriptor);
    check(!overflowing_speculative_ids_status);
    check(overflowing_speculative_ids_status.error().code == ErrorCode::InvalidModel);
    check(overflowing_speculative_ids_status.error().message == "model descriptor speculative layer IDs overflow");

    MoeModelDescriptor inconsistent_moe_descriptor = baseline_descriptor;
    inconsistent_moe_descriptor.layers.front().ffn.moe.expert_count++;
    auto inconsistent_moe_status = validate_model_descriptor(inconsistent_moe_descriptor);
    check(!inconsistent_moe_status);
    check(inconsistent_moe_status.error().code == ErrorCode::InvalidModel);
    check(inconsistent_moe_status.error().message == "layer MoE dimensions do not match the model descriptor");
    MoeModelDescriptor invalid_top_k_descriptor = baseline_descriptor;
    invalid_top_k_descriptor.layers.front().ffn.moe.top_k = invalid_top_k_descriptor.layers.front().ffn.moe.expert_count + 1;
    auto invalid_top_k_status = validate_model_descriptor(invalid_top_k_descriptor);
    check(!invalid_top_k_status);
    check(invalid_top_k_status.error().code == ErrorCode::InvalidModel);
    check(invalid_top_k_status.error().message == "invalid expert_count/top_k");
    MoeModelDescriptor inactive_attention_descriptor = baseline_descriptor;
    inactive_attention_descriptor.attention_head_count = 1;
    inactive_attention_descriptor.kv_head_count = 1;
    inactive_attention_descriptor.head_dimension = 2;
    inactive_attention_descriptor.layers.front().attention.kind = AttentionKind::None;
    inactive_attention_descriptor.layers.front().attention.head_count = 0;
    inactive_attention_descriptor.layers.front().attention.kv_head_count = 1;
    inactive_attention_descriptor.layers.front().attention.head_dimension = 3;
    inactive_attention_descriptor.layers.front().attention.flags = AttentionDescriptorSigmoidGate | std::numeric_limits<uint32_t>::max();
    check(static_cast<bool>(validate_model_descriptor(inactive_attention_descriptor)));
    auto baseline_memory = plan_model_memory(
        baseline_descriptor, options, UINT64_C(8) * 1024 * 1024 * 1024);
    auto inactive_memory = plan_model_memory(
        inactive_attention_descriptor, options, UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(baseline_memory));
    check(static_cast<bool>(inactive_memory));
    check(inactive_memory.value().estimated_dense_size == baseline_memory.value().estimated_dense_size);

    MoeModelDescriptor gated_delta_descriptor = baseline_descriptor;
    gated_delta_descriptor.attention_head_count = 1;
    gated_delta_descriptor.kv_head_count = 1;
    gated_delta_descriptor.head_dimension = 2;
    gated_delta_descriptor.layers.front().attention.kind = AttentionKind::GatedDeltaNet;
    gated_delta_descriptor.layers.front().attention.head_count = 2;
    gated_delta_descriptor.layers.front().attention.kv_head_count = 1;
    gated_delta_descriptor.layers.front().attention.head_dimension = 2;
    gated_delta_descriptor.layers.front().attention.value_head_dimension = 2;
    check(static_cast<bool>(validate_model_descriptor(gated_delta_descriptor)));
    gated_delta_descriptor.layers.front().attention.flags = AttentionDescriptorSigmoidGate;
    check(static_cast<bool>(validate_model_descriptor(gated_delta_descriptor)));

    MoeModelDescriptor known_moe_flags_descriptor = baseline_descriptor;
    known_moe_flags_descriptor.layers.front().ffn.moe.shared_expert_count = 1;
    known_moe_flags_descriptor.layers.front().ffn.moe.flags = MoeDescriptorRouterBias
                                                              | MoeDescriptorProjectionBias
                                                              | MoeDescriptorSharedExpertGate
                                                              | MoeDescriptorFileBackedExperts;
    check(static_cast<bool>(validate_model_descriptor(known_moe_flags_descriptor)));

    MoeModelDescriptor standard_flags_descriptor = gated_delta_descriptor;
    standard_flags_descriptor.layers.front().attention.kind = AttentionKind::Standard;
    standard_flags_descriptor.layers.front().attention.head_count = 1;
    standard_flags_descriptor.layers.front().attention.flags = AttentionDescriptorBias
                                                               | AttentionDescriptorSinks
                                                               | AttentionDescriptorQueryKeyNorm
                                                               | AttentionDescriptorOutputGate
                                                               | AttentionDescriptorQsa;
    check(static_cast<bool>(validate_model_descriptor(standard_flags_descriptor)));

    MoeModelDescriptor invalid_standard_flags_descriptor = standard_flags_descriptor;
    invalid_standard_flags_descriptor.layers.front().attention.flags = AttentionDescriptorSigmoidGate;
    auto invalid_standard_flags_status = validate_model_descriptor(invalid_standard_flags_descriptor);
    check(!invalid_standard_flags_status);
    check(invalid_standard_flags_status.error().code == ErrorCode::InvalidModel);
    check(invalid_standard_flags_status.error().message
          == "model descriptor layer has incompatible attention flags");

    MoeModelDescriptor latent_flags_descriptor = standard_flags_descriptor;
    latent_flags_descriptor.layers.front().attention.kind = AttentionKind::MultiHeadLatent;
    latent_flags_descriptor.layers.front().attention.flags = AttentionDescriptorSinks | AttentionDescriptorQueryKeyNorm;
    check(static_cast<bool>(validate_model_descriptor(latent_flags_descriptor)));

    MoeModelDescriptor unknown_attention_flags_descriptor = standard_flags_descriptor;
    unknown_attention_flags_descriptor.layers.front().attention.flags = std::numeric_limits<uint32_t>::max();
    auto unknown_attention_flags_status = validate_model_descriptor(unknown_attention_flags_descriptor);
    check(!unknown_attention_flags_status);
    check(unknown_attention_flags_status.error().code == ErrorCode::InvalidModel);
    check(unknown_attention_flags_status.error().message
          == "model descriptor layer has unknown attention flags");

    MoeModelDescriptor invalid_gated_delta_flags_descriptor = gated_delta_descriptor;
    invalid_gated_delta_flags_descriptor.layers.front().attention.flags = AttentionDescriptorBias
                                                                          | AttentionDescriptorSinks
                                                                          | AttentionDescriptorQueryKeyNorm
                                                                          | AttentionDescriptorOutputGate
                                                                          | AttentionDescriptorQsa;
    auto invalid_gated_delta_flags_status = validate_model_descriptor(invalid_gated_delta_flags_descriptor);
    check(!invalid_gated_delta_flags_status);
    check(invalid_gated_delta_flags_status.error().code == ErrorCode::InvalidModel);
    check(invalid_gated_delta_flags_status.error().message
          == "model descriptor layer has incompatible attention flags");

    MoeModelDescriptor invalid_latent_flags_descriptor = latent_flags_descriptor;
    invalid_latent_flags_descriptor.layers.front().attention.flags = AttentionDescriptorBias
                                                                     | AttentionDescriptorOutputGate
                                                                     | AttentionDescriptorQsa
                                                                     | AttentionDescriptorSigmoidGate;
    auto invalid_latent_flags_status = validate_model_descriptor(invalid_latent_flags_descriptor);
    check(!invalid_latent_flags_status);
    check(invalid_latent_flags_status.error().code == ErrorCode::InvalidModel);
    check(invalid_latent_flags_status.error().message
          == "model descriptor layer has incompatible attention flags");

    MoeModelDescriptor unknown_moe_flags_descriptor = baseline_descriptor;
    unknown_moe_flags_descriptor.layers.front().ffn.moe.flags = std::numeric_limits<uint32_t>::max();
    auto unknown_moe_flags_status = validate_model_descriptor(unknown_moe_flags_descriptor);
    check(!unknown_moe_flags_status);
    check(unknown_moe_flags_status.error().code == ErrorCode::InvalidModel);
    check(unknown_moe_flags_status.error().message
          == "model descriptor layer has unknown MoE flags");

    MoeModelDescriptor invalid_kind_descriptor = baseline_descriptor;
    invalid_kind_descriptor.layers.front().ffn.kind = static_cast<FfnKind>(-1);
    auto invalid_kind_status = validate_model_descriptor(invalid_kind_descriptor);
    check(!invalid_kind_status);
    check(invalid_kind_status.error().code == ErrorCode::InvalidModel);
    check(invalid_kind_status.error().message == "model descriptor layer has an invalid FFN kind");

    MoeModelDescriptor unknown_kind_descriptor = baseline_descriptor;
    unknown_kind_descriptor.layers.front().attention.kind = static_cast<AttentionKind>(-1);
    auto unknown_kind_status = validate_model_descriptor(unknown_kind_descriptor);
    check(!unknown_kind_status);
    check(unknown_kind_status.error().code == ErrorCode::InvalidModel);
    check(unknown_kind_status.error().message == "model descriptor layer has an invalid attention kind");

    MoeModelDescriptor invalid_normalization_descriptor = baseline_descriptor;
    invalid_normalization_descriptor.layers.front().ffn.moe.normalization = static_cast<RouterNormalization>(-1);
    auto invalid_normalization_status = validate_model_descriptor(invalid_normalization_descriptor);
    check(!invalid_normalization_status);
    check(invalid_normalization_status.error().code == ErrorCode::InvalidModel);
    check(invalid_normalization_status.error().message == "model descriptor layer has an invalid router normalization");

    const auto check_invalid_descriptor = [](const MoeModelDescriptor& invalid_descriptor, const char* message) {
        auto invalid_status = validate_model_descriptor(invalid_descriptor);
        check(!invalid_status);
        check(invalid_status.error().code == ErrorCode::InvalidModel);
        check(invalid_status.error().message == message);
        auto invalid_model = compile_model(invalid_descriptor, WeightMapping{});
        check(!invalid_model);
        check(invalid_model.error().code == ErrorCode::InvalidModel);
        check(invalid_model.error().message == message);
    };
    MoeModelDescriptor target_order_descriptor = baseline_descriptor;
    const LayerDescriptor baseline_layer = target_order_descriptor.layers.front();
    target_order_descriptor.layers.resize(3, baseline_layer);
    target_order_descriptor.speculative_kind = SpeculativeModelKind::DSpark;
    target_order_descriptor.speculative_layer_count = 3;
    target_order_descriptor.speculative_target_layer_ids = {2, 0, 1};
    check(static_cast<bool>(validate_model_descriptor(target_order_descriptor)));
    check(target_order_descriptor.speculative_target_layer_ids == std::vector<uint32_t>{2, 0, 1});

    MoeModelDescriptor duplicate_target_descriptor = target_order_descriptor;
    duplicate_target_descriptor.speculative_target_layer_ids = {0, 1, 0};
    check_invalid_descriptor(
        duplicate_target_descriptor,
        "speculative target layer IDs contain duplicates");

    MoeModelDescriptor out_of_range_target_descriptor = target_order_descriptor;
    out_of_range_target_descriptor.speculative_target_layer_ids = {2, 0, 3};
    check_invalid_descriptor(
        out_of_range_target_descriptor,
        "speculative target layer ID is out of range");

    MoeModelDescriptor zero_count_kind_descriptor = baseline_descriptor;
    zero_count_kind_descriptor.speculative_kind = SpeculativeModelKind::DSpark;
    check_invalid_descriptor(
        zero_count_kind_descriptor,
        "speculative fields require speculative layers");

    MoeModelDescriptor zero_count_targets_descriptor = baseline_descriptor;
    zero_count_targets_descriptor.speculative_target_layer_ids = {0};
    check_invalid_descriptor(
        zero_count_targets_descriptor,
        "speculative fields require speculative layers");

    MoeModelDescriptor missing_speculative_kind_descriptor = target_order_descriptor;
    missing_speculative_kind_descriptor.speculative_kind = SpeculativeModelKind::None;
    missing_speculative_kind_descriptor.speculative_layer_count = 1;
    missing_speculative_kind_descriptor.speculative_target_layer_ids = {0};
    check_invalid_descriptor(
        missing_speculative_kind_descriptor,
        "speculative layer kind is not configured");

    MoeModelDescriptor invalid_enum_descriptor = baseline_descriptor;
    invalid_enum_descriptor.layers.front().ffn.moe.score_function = static_cast<RouterScoreFunction>(-1);
    check_invalid_descriptor(
        invalid_enum_descriptor,
        "model descriptor layer has an invalid router score function");
    invalid_enum_descriptor = baseline_descriptor;
    invalid_enum_descriptor.layers.front().ffn.moe.activation = static_cast<ExpertActivation>(-1);
    check_invalid_descriptor(
        invalid_enum_descriptor,
        "model descriptor layer has an invalid expert activation");
    invalid_enum_descriptor = baseline_descriptor;
    invalid_enum_descriptor.layers.front().ffn.moe.layout = static_cast<ExpertLayout>(-1);
    check_invalid_descriptor(
        invalid_enum_descriptor,
        "model descriptor layer has an invalid expert layout");

    for (RouterScoreFunction score_function : {
             RouterScoreFunction::Softmax,
             RouterScoreFunction::Sigmoid,
             RouterScoreFunction::SqrtSoftplus})
    {
        MoeModelDescriptor valid_descriptor = baseline_descriptor;
        valid_descriptor.layers.front().ffn.moe.score_function = score_function;
        check(static_cast<bool>(validate_model_descriptor(valid_descriptor)));
    }
    for (ExpertActivation activation : {
             ExpertActivation::Relu,
             ExpertActivation::Silu,
             ExpertActivation::Gelu,
             ExpertActivation::ClampedSilu,
             ExpertActivation::DeepSeekSwiGlu,
             ExpertActivation::GptOssSwiGlu})
    {
        MoeModelDescriptor valid_descriptor = baseline_descriptor;
        valid_descriptor.layers.front().ffn.moe.activation = activation;
        check(static_cast<bool>(validate_model_descriptor(valid_descriptor)));
    }
    for (ExpertLayout layout : {
             ExpertLayout::UpDown,
             ExpertLayout::GateUpDown,
             ExpertLayout::PackedGateUpDown,
             ExpertLayout::InterleavedGateUpDown})
    {
        MoeModelDescriptor valid_descriptor = baseline_descriptor;
        valid_descriptor.layers.front().ffn.moe.layout = layout;
        check(static_cast<bool>(validate_model_descriptor(valid_descriptor)));
    }
    invalid_enum_descriptor = baseline_descriptor;
    invalid_enum_descriptor.hyper_connection_kind = static_cast<HyperConnectionKind>(-1);
    check_invalid_descriptor(
        invalid_enum_descriptor,
        "model descriptor has an invalid hyper-connection kind");
    for (HyperConnectionKind kind : {
             HyperConnectionKind::None,
             HyperConnectionKind::Sinkhorn,
             HyperConnectionKind::GatedResidual})
    {
        MoeModelDescriptor valid_descriptor = baseline_descriptor;
        valid_descriptor.hyper_connection_kind = kind;
        check(static_cast<bool>(validate_model_descriptor(valid_descriptor)));
    }

    MoeModelDescriptor dense_descriptor = baseline_descriptor;
    dense_descriptor.layers.front().ffn.kind = FfnKind::Dense;
    dense_descriptor.layers.front().ffn.dense_intermediate_size = dense_descriptor.intermediate_size;
    dense_descriptor.layers.front().ffn.moe.normalization = static_cast<RouterNormalization>(-1);
    dense_descriptor.layers.front().ffn.moe.flags = std::numeric_limits<uint32_t>::max();
    dense_descriptor.layers.front().ffn.moe.score_function = static_cast<RouterScoreFunction>(-1);
    dense_descriptor.layers.front().ffn.moe.activation = static_cast<ExpertActivation>(-1);
    dense_descriptor.layers.front().ffn.moe.layout = static_cast<ExpertLayout>(-1);
    auto dense_status = validate_model_descriptor(dense_descriptor);
    check(static_cast<bool>(dense_status));
    auto invalid_standard_flags_model = compile_model(
        invalid_standard_flags_descriptor, WeightMapping{});
    check(!invalid_standard_flags_model);
    check(invalid_standard_flags_model.error().code == ErrorCode::InvalidModel);
    check(invalid_standard_flags_model.error().message
          == "model descriptor layer has incompatible attention flags");
    auto invalid_moe_flags_model = compile_model(
        unknown_moe_flags_descriptor, WeightMapping{});
    check(!invalid_moe_flags_model);
    check(invalid_moe_flags_model.error().code == ErrorCode::InvalidModel);
    check(invalid_moe_flags_model.error().message
          == "model descriptor layer has unknown MoE flags");
    auto invalid_normalization_model = compile_model(invalid_normalization_descriptor, WeightMapping{});
    check(!invalid_normalization_model);
    check(invalid_normalization_model.error().code == ErrorCode::InvalidModel);
    check(invalid_normalization_model.error().message == "model descriptor layer has an invalid router normalization");
    auto dense_compiled = compile_model(dense_descriptor, WeightMapping{});
    check(!dense_compiled);
    check(dense_compiled.error().code == ErrorCode::UnsupportedModel);
    check(dense_compiled.error().message == "dense FFN decoder layers are not yet executable");

    const ExecutionGraph& graph = compiled.graph;
    check(static_cast<bool>(graph.validate()));
    check(compiled.graph.layer_plans.front().attention.kind == AttentionKind::None);
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

    const ExecutionSchedule& compiled_schedule = compiled.schedule;
    check(static_cast<bool>(compiled_schedule.validate(graph)));
    check(static_cast<bool>(compiled_schedule.node_order.size() == graph.nodes.size()));
    check(static_cast<bool>(compiled_schedule.backend_runs.size() == 1));
    check(static_cast<bool>(compiled_schedule.backend_runs.front().backend == ExecutionBackend::Cpu));
    check(static_cast<bool>(compiled_schedule.backend_runs.front().node_count == graph.nodes.size()));

    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));
    const std::array<int32_t, 1> prompt = {0};
    auto prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill));
    const SessionStatistics session_statistics = session.value()->statistics();
    check(static_cast<bool>(session_statistics.expert_batch_weight_bytes > 0));
    check(static_cast<bool>(session_statistics.expert_route_weight_bytes >= session_statistics.expert_batch_weight_bytes));
    ExecutionGraph rescheduled_graph = graph;
    GraphOption reschedule_options;
    auto rescheduled = schedule_graph(rescheduled_graph, reschedule_options);
    check(static_cast<bool>(rescheduled));
    check(static_cast<bool>(rescheduled.value().node_order == compiled_schedule.node_order));

    ExecutionGraph branch_graph;
    for (ExecutionNodeId node_id = 0; node_id < 5; ++node_id)
    {
        ExecutionNode node;
        node.id = node_id;
        node.type = ExecutionNodeType::FinalNorm;
        node.backend = ExecutionBackend::Cpu;
        node.backend_mask = ExecutionBackendCpu;
        node.name = "branch." + std::to_string(node_id);
        node.weight_inputs = {node_id};
        branch_graph.nodes.push_back(std::move(node));
    }
    branch_graph.nodes[0].dependencies = {3};
    branch_graph.nodes[2].dependencies = {1};
    branch_graph.nodes[4].dependencies = {0, 2};
    GraphOption branch_options;
    auto branch_schedule = schedule_graph(branch_graph, branch_options);
    check(static_cast<bool>(branch_schedule));
    check(static_cast<bool>(branch_schedule.value().node_order == std::vector<ExecutionNodeId>{1, 3, 2, 0, 4}));
    check(static_cast<bool>(branch_schedule.value().backend_runs.size() == 1));
    check(static_cast<bool>(branch_schedule.value().backend_runs.front().node_count == branch_graph.nodes.size()));

    ExecutionGraph duplicate_dependency_graph = branch_graph;
    duplicate_dependency_graph.nodes.back().dependencies = {0, 2, 0};
    auto duplicate_dependency = duplicate_dependency_graph.validate();
    check(!duplicate_dependency);
    check(duplicate_dependency.error().message == "execution graph contains a duplicate dependency");
    auto duplicate_schedule = schedule_graph(duplicate_dependency_graph, branch_options);
    check(!duplicate_schedule);
    check(duplicate_schedule.error().message == duplicate_dependency.error().message);

    ExecutionGraph cpu_dense_graph = graph;
    cpu_dense_graph.nodes.back().backend_mask |= ExecutionBackendVulkan;
    GraphOption cpu_dense_options;
    cpu_dense_options.available_backends = ExecutionBackendCpu | ExecutionBackendVulkan;
    cpu_dense_options.prefer_vulkan_dense = false;
    auto cpu_dense_schedule = schedule_graph(cpu_dense_graph, cpu_dense_options);
    check(static_cast<bool>(cpu_dense_schedule));
    check(static_cast<bool>(cpu_dense_graph.nodes.back().backend == ExecutionBackend::Cpu));
    check(static_cast<bool>(cpu_dense_schedule.value().backend_runs.size() == 1));

    ExecutionGraph hybrid_graph = graph;
    hybrid_graph.nodes.back().backend_mask |= ExecutionBackendVulkan;
    GraphOption scheduling_options;
    scheduling_options.available_backends = ExecutionBackendCpu | ExecutionBackendVulkan;
    auto hybrid_schedule = schedule_graph(hybrid_graph, scheduling_options);
    check(static_cast<bool>(hybrid_schedule));
    check(static_cast<bool>(hybrid_schedule.value().validate(hybrid_graph)));
    check(static_cast<bool>(hybrid_schedule.value().node_order.size() == hybrid_graph.nodes.size()));
    check(static_cast<bool>(hybrid_schedule.value().backend_runs.size() == 2));
    check(static_cast<bool>(hybrid_schedule.value().backend_runs.front().first_node == 0));
    check(static_cast<bool>(hybrid_schedule.value().backend_runs.front().node_count == hybrid_graph.nodes.size() - 1));
    check(static_cast<bool>(hybrid_schedule.value().backend_runs.back().first_node == hybrid_graph.nodes.size() - 1));
    check(static_cast<bool>(hybrid_schedule.value().backend_runs.back().node_count == 1));
    check(static_cast<bool>(hybrid_graph.nodes.back().backend == ExecutionBackend::Vulkan));

    ExecutionSchedule invalid_node_order_schedule = hybrid_schedule.value();
    std::swap(invalid_node_order_schedule.node_order.front(), invalid_node_order_schedule.node_order[1]);
    auto invalid_node_order_status = invalid_node_order_schedule.validate(hybrid_graph);
    check(!invalid_node_order_status);
    check(static_cast<bool>(invalid_node_order_status.error().code == ErrorCode::InvalidModel));
    check(static_cast<bool>(invalid_node_order_status.error().message == "execution schedule violates a node dependency"));

    ExecutionSchedule invalid_backend_run_schedule = hybrid_schedule.value();
    invalid_backend_run_schedule.backend_runs.front().backend = ExecutionBackend::Vulkan;
    auto invalid_backend_run_status = invalid_backend_run_schedule.validate(hybrid_graph);
    check(!invalid_backend_run_status);
    check(static_cast<bool>(invalid_backend_run_status.error().code == ErrorCode::InvalidModel));
    check(static_cast<bool>(invalid_backend_run_status.error().message == "execution schedule backend run disagrees with node placement"));

    ExecutionGraph invalid_dependency_graph = hybrid_graph;
    invalid_dependency_graph.nodes.back().dependencies.push_back(invalid_execution_node_id);
    auto invalid_dependency_schedule = schedule_graph(invalid_dependency_graph, scheduling_options);
    check(!invalid_dependency_schedule);
    check(static_cast<bool>(invalid_dependency_schedule.error().code == ErrorCode::InvalidModel));
    check(static_cast<bool>(invalid_dependency_schedule.error().message == "execution graph dependency is out of range"));

    for (ExecutionNodeId invalid_id : {static_cast<ExecutionNodeId>(graph.nodes.size()), invalid_execution_node_id, ExecutionNodeId{0}})
    {
        ExecutionGraph invalid_graph = hybrid_graph;
        invalid_graph.nodes.back().id = invalid_id;
        auto invalid_schedule = schedule_graph(invalid_graph, scheduling_options);
        check(!invalid_schedule);
        check(invalid_schedule.error().code == ErrorCode::InvalidModel);
        check(invalid_schedule.error().message == "execution graph node ids must be contiguous and index-aligned");
    }

    ExecutionGraph unavailable_graph = hybrid_graph;
    unavailable_graph.nodes.back().id = invalid_execution_node_id;
    unavailable_graph.nodes.front().backend_mask = 0;
    auto unavailable_schedule = schedule_graph(unavailable_graph, scheduling_options);
    check(!unavailable_schedule);
    check(unavailable_schedule.error().code == ErrorCode::UnsupportedModel);
    check(unavailable_schedule.error().message == "execution node has no available backend: " + graph.nodes.front().name);

    ExecutionGraph cyclic;
    ExecutionNode cyclic_router;
    cyclic_router.id = 0;
    cyclic_router.type = ExecutionNodeType::Router;
    cyclic_router.backend = ExecutionBackend::Cpu;
    cyclic_router.backend_mask = ExecutionBackendCpu;
    cyclic_router.layer_plan_index = 0;
    cyclic_router.name = "router";
    cyclic_router.dependencies = {1};
    ExecutionNode cyclic_combine;
    cyclic_combine.id = 1;
    cyclic_combine.type = ExecutionNodeType::Combine;
    cyclic_combine.backend = ExecutionBackend::Cpu;
    cyclic_combine.backend_mask = ExecutionBackendCpu;
    cyclic_combine.layer_plan_index = 0;
    cyclic_combine.name = "combine";
    cyclic_combine.dependencies = {0};
    cyclic.nodes = {
        std::move(cyclic_router),
        std::move(cyclic_combine),
    };
    cyclic.layer_plans.resize(1);
    GraphOption cyclic_options;
    auto invalid_schedule = schedule_graph(cyclic, cyclic_options);
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
    auto dispatch = dispatch_experts(logits, 4, options);
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
    auto weighted = dispatch_experts(weighted_logits, 1, options);
    check(static_cast<bool>(weighted));
    check(static_cast<bool>(weighted.value().assignment_count == 2));
    check(static_cast<bool>(weighted.value().batches.size() == 2));
    check(weighted.value().batches[0].routes[0].rank == 0);
    check(weighted.value().batches[1].routes[0].rank == 1);
    check_near(weighted.value().batches[0].routes[0].weight + weighted.value().batches[1].routes[0].weight, 1.0f, 1e-6f);
    ExpertDispatchPlan reusable;
    auto dispatched_into = dispatch_experts_into(weighted_logits, 1, options, reusable);
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
    dispatched_into = dispatch_experts_into(next_logits, 1, options, reusable);
    check(static_cast<bool>(dispatched_into));
    check(static_cast<bool>(reusable.batches.data() == reused_batches));
    check(static_cast<bool>(reusable.batches.front().routes.data() == reused_first_route));

    const std::vector<float> invalid_logits = {1.0f, 2.0f};
    auto invalid = dispatch_experts(invalid_logits, 1, options);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));

    const auto check_dispatch_into_matches = [&](std::span<const float> test_logits, const ExpertDispatchOptions& test_options) {
        auto expected = dispatch_experts(test_logits, 1, test_options);
        check(static_cast<bool>(expected));
        ExpertDispatchPlan actual;
        auto status = dispatch_experts_into(test_logits, 1, test_options, actual);
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
    const std::array<float, 4> tied_sigmoid_logits = {0.0f, 0.0f, -1.0f, -1.0f};
    check_dispatch_into_matches(tied_sigmoid_logits, sigmoid_options);

    // The selected scores must not already sum to one when checking normalization.
    const std::array<float, 4> sigmoid_logits = {2.0f, 1.0f, 0.0f, -1.0f};
    const float first_score = 1.0f / (1.0f + std::exp(-2.0f));
    const float second_score = 1.0f / (1.0f + std::exp(-1.0f));
    for (RouterNormalization normalization : {RouterNormalization::None, RouterNormalization::SelectedExperts})
    {
        sigmoid_options.normalization = normalization;
        auto routed = dispatch_experts(sigmoid_logits, 1, sigmoid_options);
        check(static_cast<bool>(routed));
        check(routed.value().batches.size() == 2);
        check(routed.value().batches[0].expert_id == 0);
        check(routed.value().batches[1].expert_id == 1);
        const float scale = sigmoid_options.routed_scaling_factor
                            / (normalization == RouterNormalization::SelectedExperts ? first_score + second_score : 1.0f);
        check_near(routed.value().batches[0].routes[0].weight, first_score * scale, 1e-4f);
        check_near(routed.value().batches[1].routes[0].weight, second_score * scale, 1e-4f);
        check_dispatch_into_matches(sigmoid_logits, sigmoid_options);
    }

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

    const auto check_invalid_normalization = [&](std::span<const float> test_logits, uint32_t token_count, ExpertDispatchOptions invalid_options) {
        invalid_options.normalization = static_cast<RouterNormalization>(-1);
        auto rejected = dispatch_experts(test_logits, token_count, invalid_options);
        check(!rejected);
        check(rejected.error().code == ErrorCode::InvalidArgument);
        check(rejected.error().message == "router normalization must be None or SelectedExperts");

        ExpertDispatchPlan preserved = weighted.value();
        const ExpertBatch* preserved_batches = preserved.batches.data();
        const ExpertRoute* preserved_routes = preserved.batches.front().routes.data();
        auto rejected_into = dispatch_experts_into(test_logits, token_count, invalid_options, preserved);
        check(!rejected_into);
        check(rejected_into.error().code == ErrorCode::InvalidArgument);
        check(rejected_into.error().message == "router normalization must be None or SelectedExperts");
        check(preserved.batches.data() == preserved_batches);
        check(preserved.batches.front().routes.data() == preserved_routes);
        check(preserved.assignment_count == weighted.value().assignment_count);
        check(preserved.batches.size() == weighted.value().batches.size());
        for (size_t index = 0; index < preserved.batches.size(); ++index)
        {
            const ExpertBatch& expected = weighted.value().batches[index];
            check(preserved.batches[index].expert_id == expected.expert_id);
            check(preserved.batches[index].routes.size() == expected.routes.size());
            check(preserved.batches[index].routes.front().token_index == expected.routes.front().token_index);
            check(preserved.batches[index].routes.front().rank == expected.routes.front().rank);
            check(preserved.batches[index].routes.front().weight == expected.routes.front().weight);
        }

        invalid_options.routed_scaling_factor = 0.0f;
        rejected = dispatch_experts(test_logits, token_count, invalid_options);
        check(!rejected);
        check(rejected.error().message == "routed scaling factor must be finite and positive");
        rejected_into = dispatch_experts_into(test_logits, token_count, invalid_options, preserved);
        check(!rejected_into);
        check(rejected_into.error().message == "routed scaling factor must be finite and positive");
    };
    check_invalid_normalization(weighted_logits, 1, options);
    ExpertDispatchOptions batch_options;
    batch_options.expert_count = 6;
    batch_options.top_k = 1;
    check_invalid_normalization(logits, 4, batch_options);
    check_invalid_normalization(wide_logits, 1, wide_options);
    check_invalid_normalization(duplicate_logits, 1, duplicate_options);
}

void test_deepseek_router_and_hyper_connection_kernels()
{
    ExpertDispatchOptions options;
    options.expert_count = 4;
    options.top_k = 2;
    options.score_function = RouterScoreFunction::SqrtSoftplus;
    options.routed_scaling_factor = 1.5f;
    const std::array<float, 4> selection_bias = {0.0f, 100.0f, 0.0f, 0.0f};
    options.selection_bias = selection_bias;
    const std::array<float, 4> logits = {4.0f, -4.0f, 3.0f, 2.0f};
    auto routed = dispatch_experts(logits, 1, options);
    check(static_cast<bool>(routed));
    check(routed.value().batches.size() == 2);
    check(routed.value().batches[0].expert_id == 0);
    check(routed.value().batches[1].expert_id == 1);
    check_near(routed.value().batches[0].routes[0].weight + routed.value().batches[1].routes[0].weight, 1.5f, 1e-5f);
    ExpertDispatchPlan reusable;
    auto routed_into = dispatch_experts_into(logits, 1, options, reusable);
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
    routed_into = dispatch_experts_into(logits, 1, options, reusable);
    check(static_cast<bool>(routed_into));
    check(reusable.batches.data() == reusable_batches);
    check(reusable.batches.front().routes.data() == reusable_route);

    const std::array<uint32_t, 2> explicit_experts = {3, 2};
    options.selection_bias = {};
    options.explicit_expert_ids = explicit_experts;
    routed = dispatch_experts(logits, 1, options);
    check(static_cast<bool>(routed));
    check(routed.value().batches[0].expert_id == 2);
    check(routed.value().batches[1].expert_id == 3);
    routed_into = dispatch_experts_into(logits, 1, options, reusable);
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
    auto mixed = hyper_connection_pre(
        hyper_input,
        function,
        scale,
        base,
        2,
        2,
        1e-6f,
        1e-6f,
        g_test_optimization_flags);
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

    CpuBatch four_way_branch(1, 2);
    four_way_branch.row(0)[0] = 10.0f;
    four_way_branch.row(0)[1] = 20.0f;
    CpuBatch four_way_residual(1, 8);
    const std::array<float, 8> four_way_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::copy_n(four_way_values.data(), four_way_values.size(), four_way_residual.row(0));
    CpuHyperConnectionMix four_way_mix;
    four_way_mix.post = {1.0f, 2.0f, 3.0f, 4.0f};
    four_way_mix.combine = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f};
    connected = hyper_connection_post(four_way_branch, four_way_residual, four_way_mix, 4);
    check(static_cast<bool>(connected));
    const std::array<float, 8> four_way_expected = {162.0f, 200.0f, 188.0f, 240.0f, 214.0f, 280.0f, 240.0f, 320.0f};
    for (size_t index = 0; index < four_way_expected.size(); ++index)
        check_near(connected.value().row(0)[index], four_way_expected[index], 1e-4f);

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
    std::array<float, 256> quantized_float8_input = float8_input;
    std::array<float, 256> quantized_float8_expected = float8_input;
    for (size_t block_begin = 0; block_begin < quantized_float8_expected.size();
         block_begin += 128)
    {
        const size_t block_end = std::min(
            quantized_float8_expected.size(), block_begin + 128);
        float maximum = 1e-4f;
        for (size_t index = block_begin; index < block_end; ++index)
        {
            maximum = std::max(
                maximum, std::fabs(quantized_float8_expected[index]));
        }
        const float scale = std::exp2(std::ceil(std::log2(maximum / 448.0f)));
        for (size_t index = block_begin; index < block_end; ++index)
        {
            const float normalized = std::clamp(
                quantized_float8_expected[index] / scale,
                -448.0f,
                448.0f);
            quantized_float8_expected[index] = float8_e4m3_to_float(float_to_float8_e4m3(normalized))
                                               * scale;
        }
    }
    quantize_float8_e4m3_inplace(
        quantized_float8_input.data(),
        static_cast<uint32_t>(quantized_float8_input.size()),
        128,
        true,
        g_test_optimization_flags);
    for (size_t index = 0; index < quantized_float8_input.size(); ++index)
    {
        check_near(
            quantized_float8_input[index],
            quantized_float8_expected[index],
            0.0f);
    }
    std::array<float, 256> quantized_float8_copy = {};
    quantize_float8_e4m3(
        float8_input.data(), quantized_float8_copy.data(),
        static_cast<uint32_t>(float8_input.size()),
        128,
        true,
        g_test_optimization_flags);
    for (size_t index = 0; index < quantized_float8_copy.size(); ++index)
    {
        check_near(
            quantized_float8_copy[index], quantized_float8_input[index],
            0.0f);
    }
    const float quantized_float8_reference = float8_e4m3_block_dot(
        float8_weights.data(),
        float8_scales.data(),
        quantized_float8_input.data(),
        static_cast<uint32_t>(quantized_float8_input.size()),
        128);
    check_near(
        float8_e4m3_quantized_input_dot(
            float8_weights.data(),
            float8_scales.data(),
            quantized_float8_input.data(),
            static_cast<uint32_t>(quantized_float8_input.size()),
            128,
            g_test_optimization_flags),
        quantized_float8_reference,
        1e-3f);
    std::array<float, 4> quantized_float8_row_reference = {};
    std::array<float, 4> quantized_float8_row_output = {};
    float8_e4m3_block_dot_rows4(
        float8_row_weights.data(),
        256,
        float8_scales.data(),
        quantized_float8_input.data(),
        256,
        128,
        4,
        quantized_float8_row_reference.data());
    float8_e4m3_quantized_input_dot_rows(
        float8_row_weights.data(),
        256,
        float8_scales.data(),
        quantized_float8_input.data(),
        256,
        128,
        4,
        quantized_float8_row_output.data(),
        g_test_optimization_flags);
    for (size_t row = 0; row < quantized_float8_row_output.size(); ++row)
    {
        check_near(quantized_float8_row_output[row],
                   quantized_float8_row_reference[row], 1e-3f);
    }
    std::array<float, 4 * 256> quantized_float8_batch_input = {};
    std::array<float, 4 * 4> quantized_float8_batch_expected = {};
    std::array<float, 4 * 4> quantized_float8_batch_output = {};
    for (size_t token = 0; token < 4; ++token)
    {
        for (size_t index = 0; index < 256; ++index)
        {
            quantized_float8_batch_input[token * 256 + index] = quantized_float8_input[index]
                                                                * (1.0f + static_cast<float>(token) * 0.125f);
        }
        float8_e4m3_quantized_input_dot_rows(
            float8_row_weights.data(), 256, float8_scales.data(),
            quantized_float8_batch_input.data() + token * 256, 256, 128, 4,
            quantized_float8_batch_expected.data() + token * 4,
            g_test_optimization_flags);
    }
    float8_e4m3_quantized_input_dot_rows_batch(
        float8_row_weights.data(), 256, float8_scales.data(),
        quantized_float8_batch_input.data(), 256, 256, 128, 4, 4, 4,
        quantized_float8_batch_output.data(),
        g_test_optimization_flags);
    for (size_t index = 0; index < quantized_float8_batch_output.size();
         ++index)
    {
        check_near(quantized_float8_batch_output[index],
                   quantized_float8_batch_expected[index], 1e-2f);
    }
    std::array<uint8_t, 32> repeated_float8_weights = {};
    std::array<float, 32> unit_float8_input = {};
    unit_float8_input.fill(1.0f);
    const std::array<float, 1> unit_float8_scale = {1.0f};
    for (uint32_t code = 0; code < 256; ++code)
    {
        repeated_float8_weights.fill(static_cast<uint8_t>(code));
        const float fp32_value = float8_e4m3_block_dot(
            repeated_float8_weights.data(), unit_float8_scale.data(),
            unit_float8_input.data(), 32, 32);
        const float bfloat16_value = float8_e4m3_quantized_input_dot(
            repeated_float8_weights.data(), unit_float8_scale.data(),
            unit_float8_input.data(),
            32,
            32,
            g_test_optimization_flags);
        if ((code & UINT32_C(0x7f)) == UINT32_C(0x7f))
        {
            check(std::isnan(fp32_value));
            check(std::isnan(bfloat16_value));
        }
        else
        {
            check_near(bfloat16_value, fp32_value, 1e-3f);
        }
    }
    for (uint32_t count : std::array<uint32_t, 6>{31, 32, 33, 127, 128,
                                                  129})
    {
        check_near(
            float8_e4m3_quantized_input_dot(
                float8_weights.data(), float8_scales.data(),
                quantized_float8_input.data(),
                count,
                128,
                g_test_optimization_flags),
            float8_e4m3_block_dot(
                float8_weights.data(), float8_scales.data(),
                quantized_float8_input.data(), count, 128),
            1e-3f);
    }
    std::array<uint8_t, 8 * 256> float8_rows8_weights = {};
    for (size_t row = 0; row < 8; ++row)
    {
        for (size_t index = 0; index < 256; ++index)
        {
            uint8_t encoded = static_cast<uint8_t>(index + row * 19);
            if ((encoded & UINT8_C(0x7f)) == UINT8_C(0x7f))
                encoded ^= UINT8_C(1);
            float8_rows8_weights[row * 256 + index] = encoded;
        }
    }
    for (uint32_t row_count = 1; row_count <= 8; ++row_count)
    {
        std::array<float, 8> rows8_output = {};
        float8_e4m3_quantized_input_dot_rows(
            float8_rows8_weights.data(), 256, float8_scales.data(),
            quantized_float8_input.data(), 256, 128, row_count,
            rows8_output.data(),
            g_test_optimization_flags);
        for (uint32_t row = 0; row < row_count; ++row)
        {
            check_near(
                rows8_output[row],
                float8_e4m3_block_dot(
                    float8_rows8_weights.data() + static_cast<size_t>(row) * 256,
                    float8_scales.data(), quantized_float8_input.data(), 256,
                    128),
                1e-3f);
        }
    }
    check(static_cast<bool>(std::string(float8_kernel_name()).size() > 0));
    const std::string float8_kernel_name = float8_linear_kernel_name(g_test_optimization_flags);
    check(static_cast<bool>(!float8_kernel_name.empty()));

    auto make_float8_projection = [](uint32_t seed,
                                     uint32_t output_columns) {
        TensorData matrix;
        matrix.dtype = DType::Float8E4M3;
        matrix.shape = {output_columns, 128};
        const size_t element_count = matrix.element_count();
        std::shared_ptr<uint8_t[]> storage(
            new uint8_t[element_count], std::default_delete<uint8_t[]>());
        for (size_t index = 0; index < element_count; ++index)
        {
            const float value = static_cast<float>(
                                    static_cast<int>((index * 7 + seed * 11) % 43) - 21)
                                * 0.03125f;
            storage[index] = float_to_float8_e4m3(value);
        }
        matrix.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
        matrix.mapped_size = element_count;
        matrix.quantization_scales.resize((output_columns + 127) / 128);
        for (size_t index = 0; index < matrix.quantization_scales.size();
             ++index)
        {
            matrix.quantization_scales[index] = index == 0 ? 0.5f : 2.0f;
        }
        return matrix;
    };
    const TensorData float8_gate = make_float8_projection(1, 7);
    const TensorData float8_up = make_float8_projection(2, 7);
    CpuBatch fused_input(2, 128);
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (uint32_t column = 0; column < fused_input.columns(); ++column)
        {
            fused_input.row(token_index)[column] = static_cast<float>(
                                                       static_cast<int>((token_index * 17 + column * 5) % 37)
                                                       - 18)
                                                   * 0.015625f;
        }
    }
    CpuBatch reference_up = linear_batch(
        float8_up,
        fused_input,
        g_test_optimization_flags);
    const CpuBatch reference_raw_up = reference_up;
    const CpuBatch reference_gate = linear_batch(
        float8_gate,
        fused_input,
        g_test_optimization_flags);
    for (size_t token_index = 0; token_index < reference_up.rows();
         ++token_index)
    {
        for (uint32_t column = 0; column < reference_up.columns(); ++column)
        {
            const float gate_value = reference_gate.row(token_index)[column];
            reference_up.row(token_index)[column] *= gate_value / (1.0f + std::exp(-gate_value));
        }
    }
    CpuBatch paired_gate;
    CpuBatch paired_up;
    check(float8_linear_pair_batch_into(
        float8_gate,
        float8_up,
        fused_input,
        paired_gate,
        paired_up,
        g_test_optimization_flags));
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (uint32_t column = 0; column < paired_gate.columns(); ++column)
        {
            check_near(paired_gate.row(token_index)[column],
                       reference_gate.row(token_index)[column], 1e-6f);
            check_near(paired_up.row(token_index)[column],
                       reference_raw_up.row(token_index)[column],
                       1e-6f);
        }
    }
    TensorData rms_weight;
    rms_weight.dtype = DType::Float32;
    rms_weight.shape = {128};
    rms_weight.float32_data.assign(128, 1.0f);
    const CpuBatch normalized_fused_input = rms_norm_batch(
        fused_input,
        rms_weight,
        1e-6f,
        0.0f,
        g_test_optimization_flags);
    const CpuBatch reference_rms_projection = linear_batch(
        float8_gate,
        normalized_fused_input,
        g_test_optimization_flags);
    CpuBatch fused_rms_projection;
    check(float8_linear_rms_norm_batch_into(
        float8_gate, fused_input, rms_weight, 1e-6f,
        fused_rms_projection,
        g_test_optimization_flags));
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (uint32_t column = 0;
             column < fused_rms_projection.columns(); ++column)
        {
            check_near(fused_rms_projection.row(token_index)[column],
                       reference_rms_projection.row(token_index)[column],
                       1e-6f);
        }
    }
    CpuBatch fused_output;
    check(fused_float8_gate_up_batch(
        float8_gate, float8_up, fused_input, ExpertActivation::Silu, 0.0f,
        fused_output,
        g_test_optimization_flags));
    check(fused_output.rows() == reference_up.rows());
    check(fused_output.columns() == reference_up.columns());
    for (size_t token_index = 0; token_index < reference_up.rows();
         ++token_index)
    {
        for (uint32_t column = 0; column < reference_up.columns(); ++column)
        {
            check_near(fused_output.row(token_index)[column],
                       reference_up.row(token_index)[column], 1e-6f);
        }
    }

    constexpr float deepseek_limit = 0.25f;
    CpuBatch deepseek_reference_up = linear_batch(
        float8_up,
        fused_input,
        g_test_optimization_flags);
    const CpuBatch deepseek_reference_gate = linear_batch(
        float8_gate,
        fused_input,
        g_test_optimization_flags);
    for (size_t token_index = 0;
         token_index < deepseek_reference_up.rows(); ++token_index)
    {
        for (uint32_t column = 0;
             column < deepseek_reference_up.columns(); ++column)
        {
            const float gate_value = std::min(
                deepseek_reference_gate.row(token_index)[column],
                deepseek_limit);
            const float up_value = std::clamp(
                deepseek_reference_up.row(token_index)[column],
                -deepseek_limit,
                deepseek_limit);
            deepseek_reference_up.row(token_index)[column] = up_value * gate_value / (1.0f + std::exp(-gate_value));
        }
    }
    CpuBatch deepseek_fused_output;
    check(fused_float8_gate_up_batch(
        float8_gate, float8_up, fused_input,
        ExpertActivation::DeepSeekSwiGlu, deepseek_limit,
        deepseek_fused_output,
        g_test_optimization_flags));
    for (size_t token_index = 0;
         token_index < deepseek_reference_up.rows(); ++token_index)
    {
        for (uint32_t column = 0;
             column < deepseek_reference_up.columns(); ++column)
        {
            check_near(deepseek_fused_output.row(token_index)[column],
                       deepseek_reference_up.row(token_index)[column], 1e-6f);
        }
    }

    const TensorData scale_boundary_matrix = make_float8_projection(3, 136);
    const CpuBatch scale_boundary_output = linear_batch(
        scale_boundary_matrix,
        fused_input,
        g_test_optimization_flags);
    CpuBatch quantized_boundary_input = fused_input;
    for (size_t token_index = 0;
         token_index < quantized_boundary_input.rows(); ++token_index)
    {
        quantize_float8_e4m3_inplace(
            quantized_boundary_input.row(token_index),
            128,
            128,
            true,
            g_test_optimization_flags);
    }
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (uint32_t row = 0; row < 136; ++row)
        {
            const float expected = float8_e4m3_block_dot(
                scale_boundary_matrix.float8_values().data()
                    + static_cast<size_t>(row) * 128,
                scale_boundary_matrix.quantization_scales.data() + row / 128,
                quantized_boundary_input.row(token_index), 128, 128);
            check_near(scale_boundary_output.row(token_index)[row], expected,
                       1e-3f);
        }
    }
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
    const MoeModelDescriptor& descriptor = parsed.value();
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
    Option options;
    auto memory = plan_model_memory(descriptor, options, UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(memory));
    check(memory.value().selected_mode == ExpertMemoryMode::OnDemand);
    check(memory.value().estimated_dense_size < UINT64_C(10) * 1024 * 1024 * 1024);

    auto partial_dspark = adapter.parse_model(deepseek_v4_package(R"(
        "dspark_block_size": 5,
)"));
    check(!partial_dspark);

    ModelPackage invalid = deepseek_v4_package("");
    invalid.manifest.raw_json = std::regex_replace(
        invalid.manifest.raw_json,
        std::regex(R"("o_groups"\s*:\s*8)"),
        R"("o_groups": 0)");
    auto zero_output_groups = adapter.parse_model(invalid);
    check(!zero_output_groups);
    check(zero_output_groups.error().code == ErrorCode::InvalidModel);
    check(zero_output_groups.error().message
          == "unsupported DeepSeek-V4 architectural dimensions");
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
    const std::string mtp_prefix = "__ncnn_moe_qwen3_6_mxfp4__.mtp.layers.0.experts.";
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
    const MoeModelDescriptor& descriptor = parsed.value();
    check(descriptor.model_type == "qwen3_5_moe");
    check(descriptor.norm_weight_offset == 1.0f);
    check(descriptor.layers.size() == 4);
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
    check(moe.shared_expert_count == 1);
    check(has_flag(moe.flags, MoeDescriptorSharedExpertGate));
    Option options;
    auto memory = plan_model_memory(
        descriptor,
        options,
        UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(memory));
    check(memory.value().selected_mode == ExpertMemoryMode::Eager);
    check(memory.value().expert_pair_size == 384);
    check(memory.value().estimated_expert_size == 6144);
    check(!has_flag(moe.flags, MoeDescriptorFileBackedExperts));
    Option unsupported_on_demand;
    unsupported_on_demand.expert_memory_mode = ExpertMemoryMode::OnDemand;
    check(!plan_model_memory(
        descriptor,
        unsupported_on_demand,
        UINT64_C(8) * 1024 * 1024 * 1024));

    ModelPackage invalid = qwen3_5_moe_package();
    invalid.manifest.raw_json = std::regex_replace(
        invalid.manifest.raw_json,
        std::regex(R"("mamba_ssm_dtype"\s*:\s*"float32")"),
        R"("mamba_ssm_dtype": "bfloat16")");
    check(!adapter.parse_model(invalid));

    invalid = qwen3_5_moe_package();
    invalid.manifest.raw_json = std::regex_replace(
        invalid.manifest.raw_json,
        std::regex(R"("num_key_value_heads"\s*:\s*1)"),
        R"("num_key_value_heads": 0)");
    auto zero_kv_heads = adapter.parse_model(invalid);
    check(!zero_kv_heads);
    check(zero_kv_heads.error().code == ErrorCode::InvalidModel);
    check(zero_kv_heads.error().message
          == "unsupported Qwen3 MoE architectural dimensions");

    invalid = qwen3_5_moe_package();
    invalid.manifest.raw_json = std::regex_replace(
        invalid.manifest.raw_json,
        std::regex(R"("linear_num_key_heads"\s*:\s*1)"),
        R"("linear_num_key_heads": 0)");
    auto zero_linear_key_heads = adapter.parse_model(invalid);
    check(!zero_linear_key_heads);
    check(zero_linear_key_heads.error().code == ErrorCode::InvalidModel);
    check(zero_linear_key_heads.error().message
          == "unsupported Qwen3 MoE architectural dimensions");

    for (const char* factor : {"-0.5", "1073741824", "1e38", "0", "0.75", "0.6"})
    {
        ModelPackage invalid_rotary = qwen3_5_moe_package();
        invalid_rotary.manifest.raw_json = std::regex_replace(
            invalid_rotary.manifest.raw_json,
            std::regex(R"("partial_rotary_factor"\s*:\s*0.5)"),
            std::string(R"("partial_rotary_factor": )") + factor);
        auto invalid_rotary_result = adapter.parse_model(invalid_rotary);
        check(!invalid_rotary_result);
        check(invalid_rotary_result.error().code == ErrorCode::InvalidModel);
        check(invalid_rotary_result.error().message
              == "unsupported Qwen3 MoE architectural dimensions");
    }

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

static void write_qwen4_mxfp4_test_artifact(
    ModelPackage& package,
    const std::filesystem::path& root)
{
    constexpr uint32_t layer_count = 4;
    constexpr uint32_t mtp_layer_count = 1;
    constexpr uint32_t expert_count = 4;
    constexpr uint32_t hidden_size = 32;
    constexpr uint32_t intermediate_size = 32;
    package.root = root;
    for (const auto& replacement : std::vector<std::pair<const char*, const char*>>{
             {R"("hidden_size"\s*:\s*16)", R"("hidden_size": 32)"},
             {R"("moe_intermediate_size"\s*:\s*4)", R"("moe_intermediate_size": 32)"},
             {R"("shared_expert_intermediate_size"\s*:\s*4)", R"("shared_expert_intermediate_size": 32)"},
             {R"("ple_embed_dim"\s*:\s*16)", R"("ple_embed_dim": 32)"},
         })
    {
        package.manifest.raw_json = std::regex_replace(
            package.manifest.raw_json,
            std::regex(replacement.first),
            replacement.second);
    }
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
    identity << "__ncnn_moe_qwen3_8_mxfp4__.identity.v1."
             << layer_count << '.' << mtp_layer_count << '.'
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
    header << R"({"__metadata__":{"format":"ncnn-moe-qwen3.8-mxfp4-v1"})";
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
    const auto add_bank = [&](const std::string& prefix) {
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
    };
    for (uint32_t layer_id = 0; layer_id < layer_count; ++layer_id)
    {
        add_bank(
            "__ncnn_moe_qwen3_8_mxfp4__.layers."
            + std::to_string(layer_id)
            + ".experts.");
    }
    header << '}';
    std::string encoded_header = header.str();
    encoded_header.append(
        (8 - encoded_header.size() % 8) % 8,
        ' ');
    std::ofstream artifact(
        root / "ncnn-moe-qwen3.8-mxfp4.safetensors",
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

static ModelPackage qwen4_exp_package()
{
    ModelPackage package;
    package.manifest.model_type = "qwen4_exp";
    package.manifest.raw_json = R"({
        "model_type": "qwen4_exp",
        "text_config": {
            "attention_bias": false,
            "dtype": "bfloat16",
            "eos_token_id": 127,
            "hc_count": 4,
            "hc_lowrank": 2,
            "head_dim": 8,
            "heads_per_ngram": 1,
            "hidden_act": "silu",
            "hidden_size": 16,
            "indexer_budget": 16,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 4,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 2,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ],
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 4,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 4,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 128,
            "make_ngram_vocab_size_divisible_by": 4,
            "moe_intermediate_size": 4,
            "mtp_num_hidden_layers": 1,
            "mtp": {
                "num_hidden_layers": 1,
                "rope_theta": 10000.0
            },
            "ngram_size": 3,
            "ngram_vocab_size_base": 2,
            "num_attention_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "num_key_value_heads": 1,
            "output_gate_type": "sigmoid",
            "ple_conv_kernel_size": 2,
            "ple_embed_dim": 16,
            "ple_layer_ids": [2],
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
                "partial_rotary_factor": 0.5,
                "rope_theta": 10000.0
            },
            "shared_expert_intermediate_size": 4,
            "split_ngram_parts": 2,
            "vocab_size": 128
        }
    })";
    return package;
}

void test_qwen4_exp_descriptors()
{
    Qwen4ExpModelAdapter adapter;
    auto parsed = adapter.parse_model(qwen4_exp_package());
    check(static_cast<bool>(parsed));
    const MoeModelDescriptor& descriptor = parsed.value();
    check(descriptor.model_type == "qwen4_exp");
    check(descriptor.vocabulary_size == 128);
    check(descriptor.hidden_size == 16);
    check(descriptor.layers.size() == 4);
    check(descriptor.activation_dtype == DType::BFloat16);
    check(descriptor.kv_cache_dtype == DType::BFloat16);
    check(descriptor.final_norm == NormType::None);
    check(descriptor.norm_weight_offset == 1.0f);
    check(descriptor.hyper_connection_kind
          == HyperConnectionKind::GatedResidual);
    check(descriptor.hyper_connection_multiplier == 4);
    check(descriptor.hyper_connection_low_rank == 2);
    check(descriptor.speculative_kind == SpeculativeModelKind::None);

    const AttentionDescriptor& linear = descriptor.layers[0].attention;
    check(linear.kind == AttentionKind::GatedDeltaNet);
    check(linear.head_count == 2);
    check(linear.kv_head_count == 1);
    check(linear.head_dimension == 4);
    check(linear.value_head_dimension == 4);
    check(linear.convolution_kernel_size == 2);
    check(has_flag(linear.flags, AttentionDescriptorSigmoidGate));
    check(descriptor.layers[0].pre_attention_norm == NormType::None);
    check(descriptor.layers[0].pre_ffn_norm == NormType::None);

    const AttentionDescriptor& full = descriptor.layers[3].attention;
    check(full.kind == AttentionKind::Standard);
    check(full.head_count == 2);
    check(full.kv_head_count == 1);
    check(full.head_dimension == 8);
    check(full.qk_rope_head_dimension == 4);
    check(full.index_head_count == 2);
    check(full.index_head_dimension == 4);
    check(full.index_token_budget == 16);
    check(full.index_top_k == 4);
    check(full.compression_ratio == 4);
    check(has_flag(full.flags, AttentionDescriptorQueryKeyNorm));
    check(has_flag(full.flags, AttentionDescriptorOutputGate));
    check(has_flag(full.flags, AttentionDescriptorQsa));

    const PleDescriptor& ple = descriptor.layers[1].ple;
    check(ple.enabled());
    check(ple.embedding_dimension == 16);
    check(ple.convolution_kernel_size == 2);
    check(ple.ngram_size == 3);
    check(ple.heads_per_ngram == 1);
    check(ple.embedding_shard_count == 2);
    check(ple.embedding_row_count == 8);
    check(ple.eos_token_id == 127);
    check(!descriptor.layers[0].ple.enabled());
    check(!descriptor.layers[2].ple.enabled());

    Option memory_options;
    memory_options.expert_memory_mode = ExpertMemoryMode::OnDemand;
    memory_options.expert_cache_size = 4096;
    auto memory = plan_model_memory(
        descriptor, memory_options, UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(memory));
    check(memory.value().selected_mode == ExpertMemoryMode::OnDemand);
    check(memory.value().expert_cache_size == 4096);
    MoeModelDescriptor without_ple = descriptor;
    without_ple.layers[1].ple = {};
    auto memory_without_ple = plan_model_memory(
        without_ple, memory_options, UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(memory_without_ple));
    check(memory.value().estimated_dense_size
          == memory_without_ple.value().estimated_dense_size + 3256);
    MoeModelDescriptor large_file_backed_ple = descriptor;
    large_file_backed_ple.layers[1].ple.embedding_row_count = UINT64_C(320001536);
    auto large_ple_memory = plan_model_memory(
        large_file_backed_ple, memory_options, UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(large_ple_memory));
    check(large_ple_memory.value().estimated_dense_size
          == memory.value().estimated_dense_size);

    const MoeDescriptor& moe = descriptor.layers[0].ffn.moe;
    check(moe.expert_count == 4);
    check(moe.top_k == 2);
    check(moe.shared_expert_count == 1);
    check(moe.expert_weight_dtype == DType::BFloat16);
    check(moe.layout == ExpertLayout::PackedGateUpDown);
    check(has_flag(moe.flags, MoeDescriptorSharedExpertGate));
    check(has_flag(moe.flags, MoeDescriptorFileBackedExperts));

    ModelPackage invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(
        invalid.manifest.raw_json,
        std::regex(R"("output_gate_type"\s*:\s*"sigmoid")"),
        R"("output_gate_type": "silu")");
    check(!adapter.parse_model(invalid));

    invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(
        invalid.manifest.raw_json,
        std::regex(R"("ple_layer_ids"\s*:\s*\[2\])"),
        R"("ple_layer_ids": [0])");
    check(!adapter.parse_model(invalid));

    invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(
        invalid.manifest.raw_json,
        std::regex(R"("split_ngram_parts"\s*:\s*2)"),
        R"("split_ngram_parts": 0)");
    check(!adapter.parse_model(invalid));

    invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(
        invalid.manifest.raw_json,
        std::regex(R"("num_key_value_heads"\s*:\s*1)"),
        R"("num_key_value_heads": 0)");
    check(!adapter.parse_model(invalid));

    for (const char* factor : {"-0.5", "1073741824", "1e38", "0", "0.625", "0.3"})
    {
        ModelPackage invalid_rotary = qwen4_exp_package();
        invalid_rotary.manifest.raw_json = std::regex_replace(
            invalid_rotary.manifest.raw_json,
            std::regex(R"("partial_rotary_factor"\s*:\s*0.5)"),
            std::string(R"("partial_rotary_factor": )") + factor);
        auto invalid_rotary_result = adapter.parse_model(invalid_rotary);
        check(!invalid_rotary_result);
        check(invalid_rotary_result.error().code == ErrorCode::InvalidModel);
        check(invalid_rotary_result.error().message
              == "unsupported Qwen4 Exp architectural dimensions");
    }

    invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(
        invalid.manifest.raw_json,
        std::regex(R"("ple_layer_ids"\s*:\s*\[2\])"),
        R"("ple_layer_ids": [18446744073709551616])");
    auto oversized_ple_layer_ids = adapter.parse_model(invalid);
    check(!oversized_ple_layer_ids);
    check(oversized_ple_layer_ids.error().code == ErrorCode::InvalidModel);
    check(oversized_ple_layer_ids.error().message
          == "invalid Qwen4 Exp integer array: ple_layer_ids");

    ScopedTestDirectory artifact_directory("ncnn_moe_qwen4_artifact_test_");
    ModelPackage artifact_package = qwen4_exp_package();
    write_qwen4_mxfp4_test_artifact(
        artifact_package,
        artifact_directory.path());
    auto artifact_descriptor = adapter.parse_model(artifact_package);
    check(static_cast<bool>(artifact_descriptor));
    check(artifact_descriptor.value().layers[0].ffn.moe.expert_weight_dtype
          == DType::MxFp4);
    check(artifact_descriptor.value().layers[3].ffn.moe.expert_weight_dtype
          == DType::MxFp4);
    check(has_flag(
        artifact_descriptor.value().layers[0].ffn.moe.flags,
        MoeDescriptorFileBackedExperts));

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

static void add_qwen4_mapping_bfloat16(
    WeightMapping& mapping,
    const std::string& name,
    std::vector<uint32_t> shape,
    float value = 0.0f)
{
    TensorData tensor;
    tensor.dtype = DType::BFloat16;
    tensor.shape = std::move(shape);
    tensor.bfloat16_data.assign(
        tensor.element_count(), float_to_bfloat16(value));
    check(mapping.emplace(name, std::move(tensor)).second);
}

static void add_qwen4_mapping_int64(
    WeightMapping& mapping,
    const std::string& name,
    std::vector<int64_t> values)
{
    TensorData tensor;
    tensor.dtype = DType::Int64;
    tensor.shape = {static_cast<uint32_t>(values.size())};
    tensor.int64_data = std::move(values);
    check(mapping.emplace(name, std::move(tensor)).second);
}

static WeightMapping qwen4_exp_test_mapping(const MoeModelDescriptor& descriptor)
{
    WeightMapping mapping;
    add_qwen4_mapping_bfloat16(
        mapping, "token_embedding.weight",
        {descriptor.vocabulary_size, descriptor.hidden_size});
    add_qwen4_mapping_bfloat16(
        mapping, "lm_head.weight",
        {descriptor.vocabulary_size, descriptor.hidden_size});
    const uint32_t expanded_size = descriptor.hyper_connection_multiplier
                                   * descriptor.hidden_size;
    add_qwen4_mapping_bfloat16(
        mapping, "gated_residual.head.norm.weight", {expanded_size});
    add_qwen4_mapping_bfloat16(
        mapping, "gated_residual.head.mix_down.weight",
        {descriptor.hyper_connection_low_rank, expanded_size});
    add_qwen4_mapping_bfloat16(
        mapping, "gated_residual.head.mix_up.weight",
        {expanded_size, descriptor.hyper_connection_low_rank});

    for (uint32_t layer_id = 0;
         layer_id < descriptor.layers.size();
         ++layer_id)
    {
        const std::string layer = "layers." + std::to_string(layer_id) + ".";
        for (const char* block : {"attention", "ffn"})
        {
            const std::string prefix = layer + "gated_residual." + block + ".";
            add_qwen4_mapping_bfloat16(
                mapping, prefix + "norm.weight", {expanded_size});
            add_qwen4_mapping_bfloat16(
                mapping, prefix + "mix_down.weight",
                {descriptor.hyper_connection_low_rank, expanded_size});
            add_qwen4_mapping_bfloat16(
                mapping, prefix + "mix_up.weight",
                {expanded_size, descriptor.hyper_connection_low_rank});
            add_qwen4_mapping_bfloat16(
                mapping, prefix + "inject.weight",
                {descriptor.hyper_connection_multiplier, expanded_size});
        }
        const MoeDescriptor& moe = descriptor.layers[layer_id].ffn.moe;
        add_qwen4_mapping_bfloat16(
            mapping, layer + "router.weight",
            {moe.expert_count, descriptor.hidden_size});
        add_qwen4_mapping_bfloat16(
            mapping, layer + "shared_expert.gate.weight",
            {moe.intermediate_size, descriptor.hidden_size});
        add_qwen4_mapping_bfloat16(
            mapping, layer + "shared_expert.up.weight",
            {moe.intermediate_size, descriptor.hidden_size});
        add_qwen4_mapping_bfloat16(
            mapping, layer + "shared_expert.down.weight",
            {descriptor.hidden_size, moe.intermediate_size});
        add_qwen4_mapping_bfloat16(
            mapping, layer + "shared_expert.router_gate.weight",
            {1, descriptor.hidden_size});
        for (uint32_t expert_id = 0;
             expert_id < moe.expert_count;
             ++expert_id)
        {
            const std::string expert = layer + "experts."
                                       + std::to_string(expert_id) + ".";
            add_qwen4_mapping_bfloat16(
                mapping, expert + "gate_up.weight",
                {moe.intermediate_size * 2, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, expert + "down.weight",
                {descriptor.hidden_size, moe.intermediate_size});
        }

        const AttentionDescriptor& attention = descriptor.layers[layer_id].attention;
        if (attention.kind == AttentionKind::GatedDeltaNet)
        {
            const uint32_t key_size = attention.kv_head_count
                                      * attention.head_dimension;
            const uint32_t value_size = attention.head_count
                                        * attention.value_head_dimension;
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.delta.qkv.weight",
                {key_size * 2 + value_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.delta.z.weight",
                {value_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.delta.beta.weight",
                {attention.head_count, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.delta.alpha.weight",
                {attention.head_count, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.delta.convolution.weight",
                {key_size * 2 + value_size, 1,
                 attention.convolution_kernel_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.delta.time_bias",
                {attention.head_count});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.delta.decay_log",
                {attention.head_count});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.delta.norm.weight",
                {attention.value_head_dimension});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.output.weight",
                {descriptor.hidden_size, value_size});
        }
        else
        {
            const uint32_t query_size = attention.head_count
                                        * attention.head_dimension;
            const uint32_t key_value_size = attention.kv_head_count
                                            * attention.head_dimension;
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.query.weight",
                {query_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.key.weight",
                {key_value_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.value.weight",
                {key_value_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.output.weight",
                {descriptor.hidden_size, query_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.output_gate.weight",
                {query_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.query_norm.weight",
                {attention.head_dimension});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.key_norm.weight",
                {attention.head_dimension});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.qsa.query_key.weight",
                {(attention.index_head_count + 1)
                     * attention.index_head_dimension,
                 descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.qsa.query_norm.weight",
                {attention.index_head_dimension});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "attention.qsa.key_norm.weight",
                {attention.index_head_dimension});
        }

        const PleDescriptor& ple = descriptor.layers[layer_id].ple;
        if (ple.enabled())
        {
            add_qwen4_mapping_bfloat16(
                mapping, layer + "ple.key.weight",
                {expanded_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(
                mapping, layer + "ple.value.weight",
                {descriptor.hidden_size, descriptor.hidden_size});
            for (const char* norm : {"key", "query", "convolution"})
            {
                add_qwen4_mapping_bfloat16(
                    mapping, layer + "ple." + norm + "_norm.weight",
                    {expanded_size});
            }
            add_qwen4_mapping_bfloat16(
                mapping, layer + "ple.convolution.weight",
                {expanded_size, 1, ple.convolution_kernel_size});
            add_qwen4_mapping_int64(
                mapping, layer + "ple.hash_multipliers", {1, 3, 5});
            add_qwen4_mapping_int64(
                mapping, layer + "ple.head_vocabulary_sizes", {2, 3});
            add_qwen4_mapping_int64(
                mapping, layer + "ple.head_offsets", {0, 2});
            const uint32_t head_count = (ple.ngram_size - 1)
                                        * ple.heads_per_ngram;
            const uint32_t head_dimension = ple.embedding_dimension
                                            / head_count;
            for (uint32_t shard = 0;
                 shard < ple.embedding_shard_count;
                 ++shard)
            {
                add_qwen4_mapping_bfloat16(
                    mapping,
                    layer + "ple.embedding_shard."
                        + std::to_string(shard),
                    {4, head_dimension});
            }
        }
    }
    return mapping;
}

static void make_qwen4_routed_experts_file_backed(
    WeightMapping& mapping)
{
    for (auto& item : mapping)
    {
        if (item.first.find(".experts.") == std::string::npos)
            continue;
        TensorData& tensor = item.second;
        check(tensor.dtype == DType::BFloat16);
        auto storage = std::make_shared<std::vector<uint16_t>>(
            std::move(tensor.bfloat16_data));
        tensor.bfloat16_data = {};
        tensor.mapped_data = std::shared_ptr<const uint8_t>(
            storage,
            reinterpret_cast<const uint8_t*>(storage->data()));
        tensor.mapped_size = storage->size() * sizeof(uint16_t);
    }
}

void test_qwen4_exp_compile_and_execute()
{
    Qwen4ExpModelAdapter adapter;
    auto parsed = adapter.parse_model(qwen4_exp_package());
    check(static_cast<bool>(parsed));
    WeightMapping mapping = qwen4_exp_test_mapping(parsed.value());
    make_qwen4_routed_experts_file_backed(mapping);
    CompilerOption compiler_opt;
    compiler_opt.flags |= BackendFileBackedExperts;
    auto compiled = compile_model(
        parsed.value(),
        std::move(mapping),
        HybridMode::CpuOnly,
        compiler_opt);
    if (!compiled)
    {
        throw std::runtime_error(
            "Qwen4 Exp test compilation failed: "
            + compiled.error().message);
    }
    const ExecutionTensor* delta_cache = nullptr;
    const ExecutionTensor* full_attention_cache = nullptr;
    for (const ExecutionTensor& tensor : compiled.value().graph.tensors)
    {
        if (tensor.name == "layers.0.kv_cache")
            delta_cache = &tensor;
        else if (tensor.name == "layers.3.kv_cache")
            full_attention_cache = &tensor;
    }
    check(delta_cache != nullptr);
    check(delta_cache->dtype == DType::Float32);
    check(delta_cache->shape == std::vector<uint32_t>({0, 2, 4, 4}));
    check(full_attention_cache != nullptr);
    check(full_attention_cache->dtype == DType::BFloat16);
    check(full_attention_cache->shape == std::vector<uint32_t>({0, 1, 8}));
    check(compiled.value().final_norm_weight == invalid_tensor_handle);
    check(compiled.value().gated_residual_head.norm_weight
          != invalid_tensor_handle);
    check(compiled.value().graph.layer_plans[1].ple.enabled());
    check(has_flag(
        compiled.value().graph.layer_plans[3].attention.flags,
        AttentionBlockQsa));
    check(compiled.value().graph.layer_plans[3].attention.kind
          == AttentionKind::Standard);
    check(compiled.value().graph.layer_plans[0].attention.kind
          == AttentionKind::GatedDeltaNet);
    check(has_flag(
        compiled.value().graph.layer_plans[0].attention.flags,
        AttentionBlockSigmoidGate));
    check(has_flag(
        compiled.value().graph.layer_plans[0].attention.flags,
        AttentionBlockExternalResidual));
    const uint64_t expert_pair_size = UINT64_C(3) * parsed.value().intermediate_size
                                      * parsed.value().hidden_size * sizeof(uint16_t);
    compiled.value().expert_cache = std::make_shared<ExpertCache>(
        expert_pair_size);

    CpuSessionState state;
    SessionStatistics statistics;
    const std::array<int32_t, 4> prompt = {1, 2, 9, 3};
    auto prefilled = forward_model(
        compiled.value(), prompt, statistics, state, 0);
    if (!prefilled)
    {
        throw std::runtime_error(
            "Qwen4 Exp test execution failed: "
            + prefilled.error().message);
    }
    check(prefilled.value().size() == prompt.size());
    for (const std::vector<float>& logits : prefilled.value())
    {
        check(logits.size() == parsed.value().vocabulary_size);
        for (float value : logits)
            check_near(value, 0.0f, 1e-6f);
    }
    check(state.layers[1].ple_token_history
          == std::vector<int32_t>({9, 3}));
    check(state.layers[3].qsa_index_keys.size()
          == prompt.size()
                 * parsed.value().layers[3].attention.index_head_dimension);
    check(statistics.expert_cache_misses > 0);
    check(statistics.expert_cache_bytes_read > 0);
    check(compiled.value().expert_cache->statistics().resident_size
          <= expert_pair_size);

    const uint64_t prefill_cache_misses = statistics.expert_cache_misses;
    const std::array<CpuDecodeBatchEntry, 1> entries = {{
        4,
        &statistics,
        &state,
        prompt.size(),
    }};
    auto decoded = forward_decode_batch(
        compiled.value(), entries);
    check(static_cast<bool>(decoded));
    check(decoded.value().size() == 1);
    check(state.layers[1].ple_token_history
          == std::vector<int32_t>({3, 4}));
    check(state.layers[3].qsa_index_keys.size()
          == (prompt.size() + 1)
                 * parsed.value().layers[3].attention.index_head_dimension);
    check(statistics.expert_cache_misses > prefill_cache_misses);

    MoeModelDescriptor sliding_qsa = parsed.value();
    sliding_qsa.layers[3].attention.sliding_window = 4;
    auto rejected = compile_model(
        sliding_qsa,
        qwen4_exp_test_mapping(sliding_qsa),
        HybridMode::CpuOnly);
    check(!rejected);
    check(rejected.error().code == ErrorCode::UnsupportedModel);

    WeightMapping missing_query_norm_mapping = qwen4_exp_test_mapping(parsed.value());
    missing_query_norm_mapping.erase(
        "layers.3.attention.query_norm.weight");
    missing_query_norm_mapping.erase(
        "layers.3.attention.key.weight");
    auto missing_query_norm = compile_model(
        parsed.value(),
        std::move(missing_query_norm_mapping),
        HybridMode::CpuOnly);
    check(!missing_query_norm);
    check(missing_query_norm.error().code == ErrorCode::InvalidModel);
    check(missing_query_norm.error().message
          == "missing tensor: layers.3.attention.query_norm.weight");

    WeightMapping missing_qsa_query_mapping = qwen4_exp_test_mapping(sliding_qsa);
    missing_qsa_query_mapping.erase(
        "layers.3.attention.query.weight");
    auto missing_qsa_query = compile_model(
        sliding_qsa,
        std::move(missing_qsa_query_mapping),
        HybridMode::CpuOnly);
    check(!missing_qsa_query);
    check(missing_qsa_query.error().code == ErrorCode::UnsupportedModel);
    check(missing_qsa_query.error().message
          == "QSA attention does not support a sliding-window KV cache");
}

static TensorHandle add_float_tensor(
    WeightStore& weights,
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

static TensorHandle add_bfloat16_tensor(
    WeightStore& weights,
    const std::string& name,
    std::vector<uint32_t> shape,
    const std::vector<float>& values)
{
    TensorData tensor;
    tensor.dtype = DType::BFloat16;
    tensor.shape = std::move(shape);
    tensor.bfloat16_data.reserve(values.size());
    for (float value : values)
        tensor.bfloat16_data.push_back(float_to_bfloat16(value));
    auto added = weights.add(name, std::move(tensor));
    check(static_cast<bool>(added));
    return added.value();
}

static TensorHandle add_int64_tensor(
    WeightStore& weights,
    const std::string& name,
    std::vector<uint32_t> shape,
    std::vector<int64_t> values)
{
    TensorData tensor;
    tensor.dtype = DType::Int64;
    tensor.shape = std::move(shape);
    tensor.int64_data = std::move(values);
    auto added = weights.add(name, std::move(tensor));
    check(static_cast<bool>(added));
    return added.value();
}

void test_gated_residual_kernels()
{
    WeightStore weights;
    const TensorHandle norm = add_bfloat16_tensor(
        weights, "gr_norm", {4}, {0.0f, 0.0f, 0.0f, 0.0f});
    const TensorHandle mix_down = add_bfloat16_tensor(
        weights, "gr_down", {1, 4}, {0.0f, 0.0f, 0.0f, 0.0f});
    const TensorHandle mix_up = add_bfloat16_tensor(
        weights, "gr_up", {4, 1}, {0.0f, 0.0f, 0.0f, 0.0f});
    const TensorHandle inject = add_bfloat16_tensor(
        weights, "gr_inject", {2, 4},
        {0.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 0.0f, 0.0f});

    CpuBatch input(1, 4);
    input.row(0)[0] = 1.0f;
    input.row(0)[1] = 2.0f;
    input.row(0)[2] = 3.0f;
    input.row(0)[3] = 4.0f;
    constexpr float epsilon = 1e-6f;
    auto mixed = gated_residual_pre(
        input,
        weights.at(norm),
        weights.at(mix_down),
        weights.at(mix_up),
        weights.at(inject),
        2,
        2,
        epsilon,
        1.0f,
        0);
    check(static_cast<bool>(mixed));
    const float first_scale = 1.0f / std::sqrt(2.5f + epsilon);
    const float second_scale = 1.0f / std::sqrt(12.5f + epsilon);
    check_near(
        mixed.value().reduced.row(0)[0],
        (first_scale + 3.0f * second_scale) * 0.25f,
        1e-6f);
    check_near(
        mixed.value().reduced.row(0)[1],
        (2.0f * first_scale + 4.0f * second_scale) * 0.25f,
        1e-6f);
    check(mixed.value().post.size() == 2);
    check_near(mixed.value().post[0], 1.0f, 1e-6f);
    check_near(mixed.value().post[1], 1.0f, 1e-6f);

    CpuBatch branch(1, 2);
    branch.row(0)[0] = 10.0f;
    branch.row(0)[1] = 20.0f;
    auto posted = gated_residual_post(branch, input, mixed.value(), 2);
    check(static_cast<bool>(posted));
    check_near(posted.value().row(0)[0], 11.0f, 1e-6f);
    check_near(posted.value().row(0)[1], 22.0f, 1e-6f);
    check_near(posted.value().row(0)[2], 13.0f, 1e-6f);
    check_near(posted.value().row(0)[3], 24.0f, 1e-6f);

    auto head = gated_residual_head(
        input,
        weights.at(norm),
        weights.at(mix_down),
        weights.at(mix_up),
        2,
        2,
        epsilon,
        1.0f,
        0);
    check(static_cast<bool>(head));
    check_near(
        head.value().row(0)[0], mixed.value().reduced.row(0)[0], 1e-6f);
    check_near(
        head.value().row(0)[1], mixed.value().reduced.row(0)[1], 1e-6f);
}

void test_ple_prefill_decode_continuation()
{
    WeightStore weights;
    PleBlockPlan plan;
    plan.embedding_dimension = 4;
    plan.convolution_kernel_size = 2;
    plan.ngram_size = 3;
    plan.heads_per_ngram = 1;
    plan.eos_token_id = 9;
    plan.hash_multipliers = add_int64_tensor(
        weights, "ple_multipliers", {3}, {1, 3, 5});
    plan.head_vocabulary_sizes = add_int64_tensor(
        weights, "ple_vocab", {2}, {2, 2});
    plan.head_offsets = add_int64_tensor(
        weights, "ple_offsets", {2}, {0, 2});
    plan.embedding_shards.push_back(add_bfloat16_tensor(
        weights,
        "ple_embedding",
        {4, 2},
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f,
         7.0f, 8.0f}));
    plan.key_weight = add_bfloat16_tensor(
        weights, "ple_key", {4, 4}, std::vector<float>(16, 0.0f));
    plan.value_weight = add_bfloat16_tensor(
        weights,
        "ple_value",
        {2, 4},
        {1.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f, 0.0f});
    plan.key_norm_weight = add_bfloat16_tensor(
        weights, "ple_key_norm", {4}, {0.0f, 0.0f, 0.0f, 0.0f});
    plan.query_norm_weight = add_bfloat16_tensor(
        weights, "ple_query_norm", {4}, {0.0f, 0.0f, 0.0f, 0.0f});
    plan.convolution_norm_weight = add_bfloat16_tensor(
        weights, "ple_conv_norm", {4}, {0.0f, 0.0f, 0.0f, 0.0f});
    plan.convolution_weight = add_bfloat16_tensor(
        weights,
        "ple_conv",
        {4, 1, 2},
        {1.0f, 0.0f,
         1.0f, 0.0f,
         1.0f, 0.0f,
         1.0f, 0.0f});

    const std::array<int32_t, 4> input_ids = {1, 2, 9, 3};
    CpuBatch prefill_hidden(input_ids.size(), 4);
    CpuLayerCache prefill_cache;
    check(static_cast<bool>(execute_ple_into(
        weights,
        plan,
        2,
        2,
        1e-6f,
        1.0f,
        input_ids,
        prefill_cache,
        prefill_hidden,
        0)));

    CpuBatch decode_hidden(input_ids.size(), 4);
    CpuLayerCache decode_cache;
    for (size_t row = 0; row < input_ids.size(); ++row)
    {
        CpuBatch token_hidden(1, 4);
        const std::array<int32_t, 1> token = {input_ids[row]};
        check(static_cast<bool>(execute_ple_into(
            weights,
            plan,
            2,
            2,
            1e-6f,
            1.0f,
            token,
            decode_cache,
            token_hidden,
            0)));
        std::copy_n(token_hidden.row(0), 4, decode_hidden.row(row));
    }
    for (size_t row = 0; row < input_ids.size(); ++row)
        for (size_t column = 0; column < 4; ++column)
            check_near(prefill_hidden.row(row)[column], decode_hidden.row(row)[column], 1e-6f);
    constexpr float first_gate = 0.5f;
    check_near(prefill_hidden.row(0)[0], first_gate, 1e-6f);
    check_near(prefill_hidden.row(0)[1], first_gate * 2.0f, 1e-6f);
    check_near(prefill_hidden.row(0)[2], first_gate, 1e-6f);
    check_near(prefill_hidden.row(0)[3], first_gate * 2.0f, 1e-6f);
    check(prefill_hidden.row(3)[0] > first_gate);
    check(prefill_cache.ple_token_history
          == std::vector<int32_t>({9, 3}));
    check(prefill_cache.ple_token_history
          == decode_cache.ple_token_history);
    check(prefill_cache.ple_convolution_state.size() == 12);
    check(prefill_cache.ple_convolution_state.size()
          == decode_cache.ple_convolution_state.size());
    for (size_t index = 0;
         index < prefill_cache.ple_convolution_state.size();
         ++index)
    {
        check_near(
            prefill_cache.ple_convolution_state[index],
            decode_cache.ple_convolution_state[index],
            1e-6f);
    }
}

void test_qsa_prefill_decode_continuation()
{
    WeightStore weights;
    CompiledOperatorTable operators;
    AttentionBlockPlan plan;
    plan.kind = AttentionKind::Standard;
    plan.head_count = 1;
    plan.kv_head_count = 1;
    plan.head_dimension = 2;
    plan.value_head_dimension = 2;
    plan.rope_head_dimension = 2;
    plan.max_context_length = 128;
    plan.rope_theta = 10000.0f;
    plan.norm_weight_offset = 1.0f;
    plan.index_head_count = 1;
    plan.index_head_dimension = 2;
    plan.index_top_k = 1;
    plan.index_token_budget = 2;
    plan.compression_ratio = 2;
    plan.flags = AttentionBlockQsa | AttentionBlockExternalResidual;
    const std::vector<float> identity = {
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    plan.query_weight = add_bfloat16_tensor(
        weights, "qsa_query", {2, 2}, identity);
    plan.key_weight = add_bfloat16_tensor(
        weights, "qsa_key", {2, 2}, identity);
    plan.value_weight = add_bfloat16_tensor(
        weights, "qsa_value", {2, 2}, identity);
    plan.output_weight = add_bfloat16_tensor(
        weights, "qsa_output", {2, 2}, identity);
    plan.qsa_query_key_weight = add_bfloat16_tensor(
        weights,
        "qsa_query_key",
        {4, 2},
        {1.0f, 0.0f,
         0.0f, 1.0f,
         1.0f, 0.0f,
         0.0f, 1.0f});
    plan.qsa_query_norm_weight = add_bfloat16_tensor(
        weights, "qsa_query_norm", {2}, {0.0f, 0.0f});
    plan.qsa_key_norm_weight = add_bfloat16_tensor(
        weights, "qsa_key_norm", {2}, {0.0f, 0.0f});

    CpuBatch input(6, 2);
    const float values[6][2] = {
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
        {-1.0f, 0.5f},
        {0.25f, -0.75f},
        {0.6f, 0.2f},
    };
    for (size_t row = 0; row < input.rows(); ++row)
        std::copy_n(values[row], 2, input.row(row));

    CpuLayerCache prefill_cache;
    CpuAttentionExecutionScratch prefill_scratch;
    CpuBatch prefill_output;
    check(static_cast<bool>(execute_attention_block_into(
        weights,
        operators,
        plan,
        ExecutionBackend::Cpu,
        1e-6f,
        DType::BFloat16,
        0,
        prefill_cache,
        prefill_scratch,
        input,
        prefill_output,
        0)));

    CpuLayerCache decode_cache;
    CpuAttentionExecutionScratch decode_scratch;
    CpuBatch decode_output(input.rows(), 2);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        CpuBatch token(1, 2);
        std::copy_n(input.row(row), 2, token.row(0));
        CpuBatch token_output;
        check(static_cast<bool>(execute_attention_block_into(
            weights,
            operators,
            plan,
            ExecutionBackend::Cpu,
            1e-6f,
            DType::BFloat16,
            row,
            decode_cache,
            decode_scratch,
            token,
            token_output,
            0)));
        std::copy_n(token_output.row(0), 2, decode_output.row(row));
    }
    check(prefill_cache.token_count == input.rows());
    check(prefill_cache.qsa_index_keys.size() == input.rows() * 2);
    check(prefill_cache.qsa_index_keys == decode_cache.qsa_index_keys);
    for (size_t row = 0; row < input.rows(); ++row)
        for (size_t column = 0; column < 2; ++column)
            check_near(prefill_output.row(row)[column], decode_output.row(row)[column], 1e-6f);
    check(prefill_scratch.qsa_selected_offsets.size()
          == input.rows() + 1);
    check(prefill_scratch.qsa_selected_offsets.back()
              - prefill_scratch.qsa_selected_offsets[input.rows() - 1]
          == 2);
    check(!prefill_scratch.qsa_selected_indices.empty());
    check(decode_scratch.qsa_selected_offsets.size() == 2);
    check(decode_scratch.qsa_selected_offsets.back() == 2);
    check(decode_scratch.qsa_selected_indices.size() == 2);

    AttentionBlockPlan long_plan = plan;
    long_plan.max_context_length = 262144;
    long_plan.index_top_k = 512;
    long_plan.index_token_budget = 2048;
    long_plan.compression_ratio = 4;
    constexpr uint64_t existing_tokens = 262142;
    CpuLayerCache long_cache;
    long_cache.columns = 2;
    long_cache.dtype = DType::BFloat16;
    long_cache.capacity_tokens = existing_tokens;
    long_cache.token_count = existing_tokens;
    long_cache.bfloat16_keys.assign(existing_tokens * 2, 0);
    long_cache.bfloat16_values.assign(existing_tokens * 2, 0);
    long_cache.qsa_index_keys.assign(existing_tokens * 2, 0);
    CpuAttentionExecutionScratch long_scratch;
    CpuBatch long_input(2, 2);
    long_input.row(0)[0] = 1.0f;
    long_input.row(1)[1] = 1.0f;
    CpuBatch long_output;
    check(static_cast<bool>(execute_attention_block_into(
        weights,
        operators,
        long_plan,
        ExecutionBackend::Cpu,
        1e-6f,
        DType::BFloat16,
        existing_tokens,
        long_cache,
        long_scratch,
        long_input,
        long_output,
        OptimizationCpuBf16DirectAttention)));
    check(long_cache.token_count == long_plan.max_context_length);
    check(long_scratch.qsa_selected_offsets.size() == 3);
    check(long_scratch.qsa_selected_indices.size()
          <= 2 * (long_plan.index_token_budget + long_plan.compression_ratio - 1));
    check(long_scratch.qsa_selected_offsets[1] > 0);
    check(long_scratch.qsa_selected_offsets[2]
          > long_scratch.qsa_selected_offsets[1]);
    check(long_scratch.key_cache.empty());
    check(long_scratch.value_cache.empty());
    for (size_t row = 0; row < long_output.rows(); ++row)
        for (size_t column = 0; column < long_output.columns(); ++column)
            check(std::isfinite(long_output.row(row)[column]));
}

void test_gated_delta_net_continuation()
{
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
    WeightStore weights;
    CompiledOperatorTable operators;
    AttentionBlockPlan plan;
    plan.kind = AttentionKind::GatedDeltaNet;
    plan.head_count = 1;
    plan.kv_head_count = 1;
    plan.head_dimension = 2;
    plan.value_head_dimension = 2;
    plan.convolution_kernel_size = 2;
    plan.norm_weight_offset = 1.0f;
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
    check(static_cast<bool>(execute_gated_delta_net_into(
        weights,
        operators,
        plan,
        ExecutionBackend::Cpu,
        1e-6f,
        prefill_cache,
        prefill_scratch,
        input,
        prefill_output,
        g_test_optimization_flags)));

    CpuLayerCache decode_cache;
    CpuGatedDeltaExecutionScratch decode_scratch;
    CpuBatch decode_output(2, 2);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        CpuBatch token(1, 2);
        std::copy_n(input.row(row), 2, token.row(0));
        CpuBatch token_output;
        check(static_cast<bool>(execute_gated_delta_net_into(
            weights,
            operators,
            plan,
            ExecutionBackend::Cpu,
            1e-6f,
            decode_cache,
            decode_scratch,
            token,
            token_output,
            g_test_optimization_flags)));
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
    const CompiledOperatorHandle fused_delta_handle = operators.allocate();
    operators.at_mutable(fused_delta_handle).linear = NcnnLinearOperator::create_fused(
        {
            &weights.at(plan.delta_qkv_weight),
            &weights.at(plan.delta_z_weight),
            &weights.at(plan.delta_beta_weight),
            &weights.at(plan.delta_alpha_weight),
        },
        {nullptr, nullptr, nullptr, nullptr},
        NcnnLinearDevice::Cpu,
        automatic_vulkan_device_index,
        context_instance,
        g_test_optimization_flags);
    plan.fused_delta_input_operator = fused_delta_handle;
    check(static_cast<bool>(operators.at(fused_delta_handle).linear));
    CpuLayerCache fused_cache;
    CpuGatedDeltaExecutionScratch fused_scratch;
    CpuBatch fused_output;
    check(static_cast<bool>(execute_gated_delta_net_into(
        weights,
        operators,
        plan,
        ExecutionBackend::Cpu,
        1e-6f,
        fused_cache,
        fused_scratch,
        input,
        fused_output,
        g_test_optimization_flags)));
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
    check(static_cast<bool>(execute_gated_delta_net_into(
        weights,
        operators,
        plan,
        ExecutionBackend::Cpu,
        1e-6f,
        first_row_cache,
        first_row_scratch,
        first_row_input,
        first_row_output,
        g_test_optimization_flags)));

    std::array<CpuLayerCache, 1> committed_cache;
    check(static_cast<bool>(
        begin_state_cache_transaction(committed_cache, 2)));
    auto duplicate_transaction = begin_state_cache_transaction(
        committed_cache,
        2);
    check(static_cast<bool>(!duplicate_transaction));
    check(static_cast<bool>(
        committed_cache.front().transaction.active));
    CpuGatedDeltaExecutionScratch committed_scratch;
    CpuBatch committed_output;
    check(static_cast<bool>(execute_gated_delta_net_into(
        weights,
        operators,
        plan,
        ExecutionBackend::Cpu,
        1e-6f,
        committed_cache.front(),
        committed_scratch,
        input,
        committed_output,
        g_test_optimization_flags)));
    auto committed = finish_state_cache_transaction(
        committed_cache,
        1);
    check(static_cast<bool>(committed));
    check(committed_cache.front().gated_delta_token_count == 1);
    check(committed_cache.front().gated_delta_convolution
          == first_row_cache.gated_delta_convolution);
    check(committed_cache.front().gated_delta_recurrent
          == first_row_cache.gated_delta_recurrent);
    check(static_cast<bool>(
        !committed_cache.front().transaction.active));
    check(static_cast<bool>(
        committed_cache.front().transaction.rows.empty()));
    check(static_cast<bool>(
        committed_cache.front().transaction.initial.gated_delta_recurrent.empty()));

    std::array<CpuLayerCache, 1> rolled_back_cache;
    check(static_cast<bool>(
        begin_state_cache_transaction(rolled_back_cache, 2)));
    CpuGatedDeltaExecutionScratch rolled_back_scratch;
    CpuBatch rolled_back_output;
    check(static_cast<bool>(execute_gated_delta_net_into(
        weights,
        operators,
        plan,
        ExecutionBackend::Cpu,
        1e-6f,
        rolled_back_cache.front(),
        rolled_back_scratch,
        input,
        rolled_back_output,
        g_test_optimization_flags)));
    auto rolled_back = finish_state_cache_transaction(
        rolled_back_cache,
        0);
    check(static_cast<bool>(rolled_back));
    check(rolled_back_cache.front().gated_delta_token_count == 0);
    check(rolled_back_cache.front().gated_delta_convolution.empty());
    check(rolled_back_cache.front().gated_delta_recurrent.empty());
    check(static_cast<bool>(
        !rolled_back_cache.front().transaction.active));
    check(static_cast<bool>(
        rolled_back_cache.front().transaction.rows.empty()));

    std::array<CpuLayerCache, 1> standard_cache;
    standard_cache.front().token_count = 5;
    check(static_cast<bool>(
        begin_state_cache_transaction(standard_cache, 4)));
    standard_cache.front().token_count = 9;
    record_standard_cache_transaction_rows(
        standard_cache.front(),
        4);
    auto standard_committed = finish_state_cache_transaction(
        standard_cache,
        2);
    check(static_cast<bool>(standard_committed));
    check(standard_cache.front().token_count == 7);

    std::array<CpuLayerCache, 1> sliding_cache;
    sliding_cache.front().token_count = 4;
    sliding_cache.front().capacity_tokens = 4;
    sliding_cache.front().first_slot = 1;
    auto sliding_transaction = begin_state_cache_transaction(
        sliding_cache,
        2);
    check(static_cast<bool>(!sliding_transaction));
    check(static_cast<bool>(
        sliding_transaction.error().code == ErrorCode::UnsupportedModel));
    check(static_cast<bool>(
        !sliding_cache.front().transaction.active));

    CpuLayerCache poisoned_attention_cache;
    poisoned_attention_cache.vulkan_attention_state_unknown = true;
    CpuAttentionExecutionScratch poisoned_attention_scratch;
    CpuBatch poisoned_attention_hidden(1, 1);
    CpuBatch poisoned_attention_output;
    WeightStore poisoned_attention_weights;
    CompiledOperatorTable poisoned_attention_operators;
    AttentionBlockPlan poisoned_attention_plan;
    poisoned_attention_plan.kind = AttentionKind::Standard;
    auto poisoned_attention = execute_attention_block_into(
        poisoned_attention_weights,
        poisoned_attention_operators,
        poisoned_attention_plan,
        ExecutionBackend::Cpu,
        1e-6f,
        DType::Float32,
        0,
        poisoned_attention_cache,
        poisoned_attention_scratch,
        poisoned_attention_hidden,
        poisoned_attention_output,
        g_test_optimization_flags);
    check(static_cast<bool>(!poisoned_attention));
    check(static_cast<bool>(
        poisoned_attention.error().code == ErrorCode::InternalError));
}

static MoeModelDescriptor gpt_oss_memory_descriptor(uint32_t layer_count, uint32_t expert_count)
{
    MoeModelDescriptor descriptor;
    descriptor.model_type = "gpt_oss";
    descriptor.vocabulary_size = 201088;
    descriptor.hidden_size = 2880;
    descriptor.intermediate_size = 2880;
    descriptor.attention_head_count = 64;
    descriptor.kv_head_count = 8;
    descriptor.head_dimension = 64;
    descriptor.expert_count = expert_count;
    descriptor.experts_per_token = 4;
    descriptor.activation_dtype = DType::BFloat16;
    descriptor.kv_cache_dtype = DType::BFloat16;
    descriptor.layers.resize(layer_count);
    for (LayerDescriptor& layer : descriptor.layers)
    {
        layer.attention.kind = AttentionKind::Standard;
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
    return descriptor;
}

void test_automatic_expert_memory_planning()
{
    static constexpr uint64_t gibibyte = 1024ull * 1024ull * 1024ull;
    Option options;
    const uint64_t physical_memory = 32 * gibibyte;

    const MoeModelDescriptor small = gpt_oss_memory_descriptor(24, 32);
    auto small_plan = plan_model_memory(small, options, physical_memory);
    check(static_cast<bool>(small_plan));
    check(static_cast<bool>(small_plan.value().requested_mode == ExpertMemoryMode::Auto));
    check(static_cast<bool>(small_plan.value().selected_mode == ExpertMemoryMode::Eager));
    check(static_cast<bool>(small_plan.value().estimated_expert_size < 11 * gibibyte));
    check(static_cast<bool>(small_plan.value().estimated_cpu_packed_expert_size == 0));
    check(static_cast<bool>(small_plan.value().estimated_expert_resident_size == small_plan.value().estimated_expert_size));

    const MoeModelDescriptor large = gpt_oss_memory_descriptor(36, 128);
    auto large_plan = plan_model_memory(large, options, physical_memory);
    check(static_cast<bool>(large_plan));
    check(static_cast<bool>(large_plan.value().requested_mode == ExpertMemoryMode::Auto));
    check(static_cast<bool>(large_plan.value().selected_mode == ExpertMemoryMode::OnDemand));
    check(static_cast<bool>(large_plan.value().host_memory_budget == 24 * gibibyte));
    check(static_cast<bool>(large_plan.value().estimated_dense_size == 4334742144ull));
    check(static_cast<bool>(large_plan.value().expert_pair_size == 13219200ull));
    check(static_cast<bool>(large_plan.value().expert_pair_resident_size == large_plan.value().expert_pair_size));
    check(static_cast<bool>(large_plan.value().estimated_expert_size == 60914073600ull));
    check(static_cast<bool>(large_plan.value().estimated_cpu_packed_expert_size == 0));
    check(static_cast<bool>(large_plan.value().estimated_expert_resident_size == large_plan.value().estimated_expert_size));
    check(static_cast<bool>(large_plan.value().expert_cache_size == 20 * gibibyte - large_plan.value().estimated_dense_size));
    check(static_cast<bool>(large_plan.value().expert_cache_size >= large_plan.value().minimum_active_expert_size));

    Option invalid_mode;
    invalid_mode.expert_memory_mode = static_cast<ExpertMemoryMode>(-1);
    auto invalid_mode_plan = plan_model_memory(large, invalid_mode, physical_memory);
    check(static_cast<bool>(!invalid_mode_plan));
    check(static_cast<bool>(invalid_mode_plan.error().code == ErrorCode::InvalidArgument));
    check(static_cast<bool>(invalid_mode_plan.error().message == "expert memory mode is invalid"));
    invalid_mode.expert_cache_size = large_plan.value().expert_cache_size;
    invalid_mode_plan = plan_model_memory(large, invalid_mode, physical_memory);
    check(static_cast<bool>(!invalid_mode_plan));
    check(static_cast<bool>(invalid_mode_plan.error().code == ErrorCode::InvalidArgument));
    check(static_cast<bool>(invalid_mode_plan.error().message == "expert memory mode is invalid"));

    auto available_limited_plan = plan_model_memory(
        large,
        options,
        physical_memory,
        false,
        12 * gibibyte);
    check(static_cast<bool>(available_limited_plan));
    check(static_cast<bool>(available_limited_plan.value().host_memory_budget == 10 * gibibyte));
    check(static_cast<bool>(available_limited_plan.value().expert_cache_size
                            == 10 * gibibyte - available_limited_plan.value().estimated_dense_size));

    MoeModelDescriptor qnk = gpt_oss_memory_descriptor(1, 24);
    qnk.vocabulary_size = 128;
    qnk.hidden_size = 4096;
    qnk.intermediate_size = 4096;
    qnk.layers.front().ffn.moe.intermediate_size = 4096;
    qnk.layers.front().ffn.moe.expert_weight_dtype = DType::Q4K;
    Option qnk_options;
    qnk_options.host_memory_budget = 3 * gibibyte;
    auto qnk_raw_plan = plan_model_memory(qnk, qnk_options, 4 * gibibyte);
    check(static_cast<bool>(qnk_raw_plan));
    check(static_cast<bool>(qnk_raw_plan.value().estimated_cpu_packed_expert_size == 0));
    auto qnk_packed_plan = plan_model_memory(
        qnk,
        qnk_options,
        4 * gibibyte,
        false,
        0,
        true);
    check(static_cast<bool>(!qnk_packed_plan));
    check(static_cast<bool>(qnk_packed_plan.error().code == ErrorCode::InvalidArgument));

    auto available_plan = plan_model_memory(
        large,
        options,
        physical_memory,
        false,
        16 * gibibyte);
    check(static_cast<bool>(available_plan));
    check(static_cast<bool>(
        available_plan.value().available_memory_size == 16 * gibibyte));
    check(static_cast<bool>(
        available_plan.value().host_memory_budget == 14 * gibibyte));

    Option undersized;
    undersized.expert_cache_size = 32 * 1024 * 1024;
    auto invalid = plan_model_memory(large, undersized, physical_memory);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));

    Option over_budget;
    over_budget.host_memory_budget = 33 * gibibyte;
    invalid = plan_model_memory(large, over_budget, physical_memory);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));

    Option oversized_cache;
    oversized_cache.host_memory_budget = 24 * gibibyte;
    oversized_cache.expert_cache_size = 21 * gibibyte;
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

    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    SamplingOptions greedy_options;
    greedy_options.temperature = 0.0f;
    auto greedy = session.value()->sample(logits, greedy_options);
    check(static_cast<bool>(greedy));
    check(static_cast<bool>(greedy.value().token_id == 2));
    check_near(greedy.value().probability, 1.0f, 1e-6f);

    logits = {3.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    greedy = session.value()->sample(logits, greedy_options);
    check(static_cast<bool>(greedy));
    check(static_cast<bool>(greedy.value().token_id == 0));

    logits = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    greedy = session.value()->sample(logits, greedy_options);
    check(static_cast<bool>(!greedy));
    check(static_cast<bool>(greedy.error().code == ErrorCode::InvalidArgument));

    logits = {1.0f, 2.0f, 3.0f};
    SamplingOptions top_k_options;
    top_k_options.top_k = 1;
    auto top_k = session.value()->sample(logits, top_k_options);
    check(static_cast<bool>(top_k));
    check(static_cast<bool>(top_k.value().token_id == 2));

    logits = {-10.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f};
    SamplingOptions bounded_options;
    bounded_options.top_k = 2;
    bounded_options.top_p = 0.7f;
    auto bounded = session.value()->sample(logits, bounded_options);
    check(static_cast<bool>(bounded));
    check(static_cast<bool>(bounded.value().token_id == 1));
    check_near(bounded.value().probability, 1.0f, 1e-6f);

    bounded_options.top_p = 1.0f;
    bounded_options.min_p = 0.2f;
    bounded = session.value()->sample(logits, bounded_options);
    check(static_cast<bool>(bounded));
    check(static_cast<bool>(bounded.value().token_id == 1 || bounded.value().token_id == 2));

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
    const SessionMetrics metrics = session.value()->metrics();
    check(static_cast<bool>(!metrics.timing.active));
    check(static_cast<bool>(metrics.timing.input_tokens == 1));
    check(static_cast<bool>(metrics.timing.output_tokens == 3));
    check(static_cast<bool>(metrics.timing.prompt_elapsed_microseconds.has_value()));
    check(static_cast<bool>(metrics.timing.generation_elapsed_microseconds.has_value()));
    check(static_cast<bool>(metrics.timing.prompt_tokens_per_second.has_value()));
    check(static_cast<bool>(metrics.timing.generation_tokens_per_second.has_value()));
    check(static_cast<bool>(metrics.timing.ttft_microseconds.has_value()));
    check(static_cast<bool>(metrics.timing.tpot_microseconds.has_value()));
    check(static_cast<bool>(metrics.timing.decode_tokens_per_second.has_value()));
    check(static_cast<bool>(metrics.timing.generation_tokens_per_second == metrics.timing.decode_tokens_per_second));
    check(static_cast<bool>(metrics.generation.prefill_tokens == 1));
    check(static_cast<bool>(metrics.cumulative.prefill_tokens >= metrics.generation.prefill_tokens));

    auto stopped_session = runtime.create_session(model.value(), session_options);
    check(static_cast<bool>(stopped_session));
    auto stopped = stopped_session.value()->generate(std::vector<int32_t>{0}, generation_options, [](const StreamToken&) { return false; });
    check(static_cast<bool>(stopped));
    check(static_cast<bool>(stopped.value().stopped_by_callback));
    check(static_cast<bool>(stopped.value().tokens.size() == 1));
    check(static_cast<bool>(stopped_session.value()->sequence_length() == 1));
}

void test_generation_callback_errors()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));
    GenerationOptions opt;
    opt.max_new_tokens = 2;
    opt.sampling.temperature = 0.0f;
    opt.use_speculative = false;

    for (bool throw_from_decoder : {true, false})
    {
        check(static_cast<bool>(session.value()->reset()));
        bool callback_called = false;
        bool caught = false;
        try
        {
            auto result = session.value()->generate(
                std::vector<int32_t>{0}, opt,
                [&](const StreamToken&) -> bool {
                    callback_called = true;
                    check(session.value()->metrics().timing.active);
                    throw std::runtime_error("callback failed");
                },
                [&](int32_t) -> std::string {
                    check(session.value()->metrics().timing.active);
                    if (throw_from_decoder)
                        throw std::runtime_error("decoder failed");
                    return "token";
                });
            (void)result;
        }
        catch (const std::runtime_error& error)
        {
            caught = true;
            check(std::string(error.what()) == (throw_from_decoder ? "decoder failed" : "callback failed"));
        }
        check(caught);
        check(callback_called == !throw_from_decoder);
        const SessionMetrics metrics = session.value()->metrics();
        check(!metrics.timing.active);
        check(metrics.timing.output_tokens == 1);
        check(metrics.timing.ttft_microseconds.has_value());
        check(session.value()->sequence_length() == 1);
    }

    check(static_cast<bool>(session.value()->reset()));
    auto modified = session.value()->generate(std::vector<int32_t>{0}, opt, [&](const StreamToken& token) {
        check(static_cast<bool>(session.value()->decode(token.token_id)));
        return true;
    });
    check(!modified);
    check(modified.error().message == "generation callback modified the Session");
    check(!session.value()->metrics().timing.active);

    check(static_cast<bool>(session.value()->reset()));
    auto invalid = session.value()->generate(std::vector<int32_t>{-1}, opt);
    check(!invalid);
    const SessionMetrics failed_metrics = session.value()->metrics();
    check(!failed_metrics.timing.active);
    check(failed_metrics.timing.output_tokens == 0);
    check(!failed_metrics.timing.ttft_microseconds.has_value());
    auto resumed = session.value()->generate(std::vector<int32_t>{0}, opt);
    check(static_cast<bool>(resumed));
    check(resumed.value().tokens.size() == opt.max_new_tokens);
    check(!session.value()->metrics().timing.active);
}

void test_manifest_readers()
{
    const std::string json = R"({"width":4294967295,"zero":0,"positive":12.5,"negative":-2.5,"exponent":1.5e+2,"nested":{"kind":"first","value":7,"nested_scale":3.25,"nested_flag":false},"kind":"second","nested_scale":8.5,"nested_flag":true,"flag":true,"scale":-1.25e-3})";
    auto width = read_manifest_uint32(json, "width");
    check(static_cast<bool>(width));
    check(width.value() == std::numeric_limits<uint32_t>::max());
    auto zero = read_manifest_uint32(json, "zero");
    check(static_cast<bool>(zero));
    check(zero.value() == 0);
    auto nested = read_manifest_uint32(json, "value");
    check(static_cast<bool>(nested));
    check(nested.value() == 7);
    auto kind = read_manifest_string(json, "kind");
    check(static_cast<bool>(kind));
    check(kind.value() == "first");
    auto positive = read_manifest_float(json, "positive");
    check(static_cast<bool>(positive));
    check_near(positive.value(), 12.5f, 1e-6f);
    auto negative = read_manifest_float(json, "negative");
    check(static_cast<bool>(negative));
    check_near(negative.value(), -2.5f, 1e-6f);
    auto exponent = read_manifest_float(json, "exponent");
    check(static_cast<bool>(exponent));
    check_near(exponent.value(), 150.0f, 1e-6f);
    auto nested_scale = read_manifest_float(json, "nested_scale");
    check(static_cast<bool>(nested_scale));
    check_near(nested_scale.value(), 3.25f, 1e-6f);
    auto nested_flag = read_manifest_bool(json, "nested_flag");
    check(static_cast<bool>(nested_flag));
    check(!nested_flag.value());
    auto flag = read_manifest_bool(json, "flag");
    check(static_cast<bool>(flag));
    check(flag.value());

    for (const char* prefix : {"", "DeepSeek-V4 ", "Qwen3 MoE "})
    {
        auto missing = read_manifest_uint32(json, "missing", prefix);
        check(!missing);
        check(missing.error().code == ErrorCode::InvalidModel);
        check(missing.error().message == std::string(prefix) + "manifest is missing integer field: missing");
        auto overflow = read_manifest_uint32(R"({"width":4294967296})", "width", prefix);
        check(!overflow);
        check(overflow.error().message == std::string(prefix) + "manifest integer is out of range: width");
        auto invalid = read_manifest_uint32(R"({"width":999999999999999999999999999999})", "width", prefix);
        check(!invalid);
        check(invalid.error().message == "invalid " + std::string(prefix) + "integer field: width");
        auto empty = read_manifest_string(R"({"kind":""})", "kind", prefix);
        check(!empty);
        check(empty.error().message == std::string(prefix) + "manifest is missing string field: kind");
        auto missing_float = read_manifest_float(json, "missing_float", prefix);
        check(!missing_float);
        check(missing_float.error().message == std::string(prefix) + "manifest is missing numeric field: missing_float");
        auto invalid_float = read_manifest_float(R"({"scale":1e999})", "scale", prefix);
        check(!invalid_float);
        check(invalid_float.error().message == "invalid " + std::string(prefix) + "numeric field: scale");
        auto missing_bool = read_manifest_bool(json, "missing_bool", prefix);
        check(!missing_bool);
        check(missing_bool.error().message == std::string(prefix) + "manifest is missing boolean field: missing_bool");
    }
    check_near(optional_manifest_float(json, "scale", 2.0f), -0.00125f, 1e-8f);
    check_near(optional_manifest_float(json, "missing", 2.0f), 2.0f, 0.0f);
    check_near(optional_manifest_float(R"({"scale":"invalid"})", "scale", 2.0f), 2.0f, 0.0f);
    check_near(optional_manifest_float(R"({"scale":1e999})", "scale", 2.0f), 2.0f, 0.0f);
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
        Option options;
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
        Option options;
        options.expert_memory_mode = ExpertMemoryMode::Eager;
        options.expert_gpu_cache_size = 64 * 1024 * 1024;
        auto model = runtime.load_model(package.path(), options);
        check(static_cast<bool>(!model));
        check(static_cast<bool>(model.error().code == ErrorCode::InvalidArgument));
    }

    {
        TemporaryModelPackage package;
        TestRuntime runtime;
        Option options;
        options.expert_memory_mode = ExpertMemoryMode::Eager;
        options.expert_gpu_victim_cache_size = 64 * 1024 * 1024;
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
    check(static_cast<bool>(has_flag(runtime.info().flags, RuntimeCpu)));
    Option invalid_device_options;
    invalid_device_options.vulkan_device_index = static_cast<uint32_t>(runtime.info().gpu_infos.size());
    auto invalid_device_model = runtime.load_model(package.path(), invalid_device_options);
    check(static_cast<bool>(!invalid_device_model));
    check(static_cast<bool>(invalid_device_model.error().code == ErrorCode::InvalidArgument));
    if (!runtime.info().gpu_infos.empty())
    {
        Option duplicate_devices;
        duplicate_devices.vulkan_device_indices = {
            0,
            0,
        };
        auto duplicate_device_model = runtime.load_model(package.path(), duplicate_devices);
        check(static_cast<bool>(!duplicate_device_model));
        check(static_cast<bool>(duplicate_device_model.error().code == ErrorCode::InvalidArgument));
    }

    Option automatic_options;
    automatic_options.hybrid_mode = HybridMode::Auto;
    const uint32_t automatic_device_index = runtime.info().default_gpu_index;
    const bool automatic_uses_vulkan = has_flag(runtime.info().flags, RuntimeVulkanCpu) && automatic_device_index < runtime.info().gpu_infos.size()
                                       && runtime.info().gpu_infos[automatic_device_index].type != VulkanDeviceType::Cpu;
    auto automatic_model = runtime.load_model(package.path(), automatic_options);
    check(static_cast<bool>(automatic_model));
    check(static_cast<bool>(runtime.synchronize_model_caches(automatic_model.value())));
    check(static_cast<bool>(automatic_model.value()->hybrid_mode() == (automatic_uses_vulkan ? HybridMode::HybridExperts : HybridMode::CpuOnly)));
    check(static_cast<bool>(automatic_model.value()->vulkan_device_index() == (automatic_uses_vulkan ? automatic_device_index : automatic_vulkan_device_index)));
    const CompiledModel& automatic_compiled = model_compiled(*automatic_model.value());
    const EffectiveOption& automatic_effective = automatic_compiled.opt;
    check(static_cast<bool>(automatic_effective.hybrid_mode == automatic_model.value()->hybrid_mode()));
    check(static_cast<bool>(automatic_compiled.memory_plan.requested_mode == automatic_options.expert_memory_mode));
    check(static_cast<bool>(automatic_compiled.memory_plan.selected_mode != ExpertMemoryMode::Auto));
    check(static_cast<bool>(automatic_compiled.memory_plan.host_memory_budget != 0));
    check(static_cast<bool>(automatic_compiled.memory_plan.expert_cache_size == 0
                            || automatic_compiled.memory_plan.expert_cache_size >= automatic_compiled.memory_plan.minimum_active_expert_size));
    check(static_cast<bool>(!has_flag(
        automatic_effective.optimization_flags,
        OptimizationCpuPackedWeights)));
    check(static_cast<bool>(automatic_effective.num_concurrent_sessions == automatic_options.num_concurrent_sessions));
    check(static_cast<bool>(automatic_effective.vulkan_device_indices == automatic_model.value()->vulkan_device_indices()));
    auto automatic_session = runtime.create_session(automatic_model.value());
    check(static_cast<bool>(automatic_session));
    const std::vector<int32_t> packed_prompt = {0, 1, 2, 3};
    auto automatic_prefill = automatic_session.value()->prefill(packed_prompt);
    check(static_cast<bool>(automatic_prefill));
    if (automatic_uses_vulkan)
    {
        Option selected_device_options;
        selected_device_options.hybrid_mode = HybridMode::HybridExperts;
        selected_device_options.vulkan_device_index = runtime.info().default_gpu_index;
        auto selected_device_model = runtime.load_model(package.path(), selected_device_options);
        check(static_cast<bool>(selected_device_model));
        check(static_cast<bool>(selected_device_model.value()->vulkan_device_index() == selected_device_options.vulkan_device_index));

        if (runtime.info().gpu_infos.size() >= 2 && runtime.info().gpu_infos[0].type != VulkanDeviceType::Cpu && runtime.info().gpu_infos[1].type != VulkanDeviceType::Cpu)
        {
            Option multi_device_options;
            multi_device_options.hybrid_mode = HybridMode::HybridExperts;
            multi_device_options.vulkan_device_indices = {
                0,
                1,
            };
            auto multi_device_model = runtime.load_model(package.path(), multi_device_options);
            check(static_cast<bool>(multi_device_model));
            check(static_cast<bool>(multi_device_model.value()->vulkan_device_indices() == std::vector<uint32_t>({0})));
            const CompiledModel& compiled = model_compiled(*multi_device_model.value());
            check(static_cast<bool>(
                compiled.graph.layer_plans.front().vulkan_device_index
                < runtime.info().gpu_infos.size()));
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

    Option cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto cpu_model = runtime.load_model(package.path(), cpu_options);
    check(static_cast<bool>(cpu_model));
    check(static_cast<bool>(cpu_model.value()->vulkan_device_index() == automatic_vulkan_device_index));
    auto cpu_session = runtime.create_session(cpu_model.value());
    check(static_cast<bool>(cpu_session));
    auto cpu_prefill = cpu_session.value()->prefill(packed_prompt);
    check(static_cast<bool>(cpu_prefill));
    check(static_cast<bool>(cpu_prefill.value().logits.size() == automatic_prefill.value().logits.size()));
    for (size_t index = 0; index < cpu_prefill.value().logits.size(); ++index)
    {
        check_near(automatic_prefill.value().logits[index], cpu_prefill.value().logits[index], 1e-4f);
    }

    Option hybrid_options;
    hybrid_options.hybrid_mode = HybridMode::HybridExperts;
    auto hybrid_model = runtime.load_model(package.path(), hybrid_options);
    if (has_flag(runtime.info().flags, RuntimeVulkanCpu))
    {
        check(static_cast<bool>(hybrid_model));
        check(static_cast<bool>(hybrid_model.value()->hybrid_mode() == HybridMode::HybridExperts));

        AttentionPackage attention_package;
        Option attention_options = hybrid_options;
        attention_options.optimization_flags |= OptimizationVulkanDecodeSdpa;
        auto attention_model = runtime.load_model(
            attention_package.path(),
            attention_options);
        check(static_cast<bool>(attention_model));
        auto attention_session = runtime.create_session(attention_model.value());
        check(static_cast<bool>(attention_session));
        const std::vector<int32_t> attention_prompt = {0, 1, 0, 1};
        auto attention_prefill = attention_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(attention_prefill));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_attention_blocks == 1));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_attention_qkv_rope_fusions == 1));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_attention_device_rope_fusions == 1));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_attention_qkv_ring_fusions == 0));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_attention_decode_sdpa_fusions == 0));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_linear_dispatches == 3));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_compute_submissions == 2));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_batch_uploads == 2));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_batch_downloads == 2));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_auxiliary_uploads == 1));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_auxiliary_upload_bytes > 0));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_staging_slot_resizes + attention_session.value()->statistics().vulkan_staging_slot_reuses == 5));
        check(static_cast<bool>(attention_session.value()->statistics().vulkan_staging_slot_acquisitions == 2));
        check(static_cast<bool>(attention_session.value()->statistics().kv_cache_logical_size > 0));
        check(static_cast<bool>(attention_session.value()->statistics().kv_cache_allocated_size >= attention_session.value()->statistics().kv_cache_logical_size));

        auto cpu_attention_model = runtime.load_model(attention_package.path(), cpu_options);
        check(static_cast<bool>(cpu_attention_model));
        auto cpu_attention_session = runtime.create_session(cpu_attention_model.value());
        check(static_cast<bool>(cpu_attention_session));
        auto cpu_attention_prefill = cpu_attention_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(cpu_attention_prefill));
        check(static_cast<bool>(cpu_attention_prefill.value().logits.size() == attention_prefill.value().logits.size()));
        for (size_t index = 0; index < cpu_attention_prefill.value().logits.size(); ++index)
        {
            check_near(attention_prefill.value().logits[index], cpu_attention_prefill.value().logits[index], 1e-4f);
        }

        {
            Option control_options = hybrid_options;
            control_options.optimization_flags &= ~OptimizationVulkanDeviceRope;
            auto control_attention_model = runtime.load_model(attention_package.path(), control_options);
            check(static_cast<bool>(control_attention_model));
            auto control_attention_session = runtime.create_session(control_attention_model.value());
            check(static_cast<bool>(control_attention_session));
            auto control_attention_prefill = control_attention_session.value()->prefill(attention_prompt);
            check(static_cast<bool>(control_attention_prefill));
            const SessionStatistics& control_statistics = control_attention_session.value()->statistics();
            check(static_cast<bool>(
                control_statistics.vulkan_attention_device_rope_fusions == 0));
            check(static_cast<bool>(control_statistics.vulkan_auxiliary_uploads == 3));
            check(static_cast<bool>(
                control_attention_prefill.value().logits.size()
                == attention_prefill.value().logits.size()));
            for (size_t index = 0;
                 index < attention_prefill.value().logits.size();
                 ++index)
            {
                check_near(
                    control_attention_prefill.value().logits[index],
                    attention_prefill.value().logits[index],
                    1e-4f);
            }
        }

        auto check_attention_cache_isolation =
            [&](const std::filesystem::path& package_path) {
                Option disabled_attention_options = hybrid_options;
                disabled_attention_options.optimization_flags &= ~OptimizationVulkanAttention;
                auto promotion_model = runtime.load_model(
                    package_path,
                    disabled_attention_options);
                auto promotion_cpu_model = runtime.load_model(package_path, cpu_options);
                check(static_cast<bool>(promotion_model));
                check(static_cast<bool>(promotion_cpu_model));
                auto promotion_session = runtime.create_session(
                    promotion_model.value());
                auto promotion_cpu_session = runtime.create_session(
                    promotion_cpu_model.value());
                check(static_cast<bool>(promotion_session));
                check(static_cast<bool>(promotion_cpu_session));
                check(static_cast<bool>(
                    promotion_session.value()->prefill(
                        attention_prompt)));
                check(static_cast<bool>(
                    promotion_cpu_session.value()->prefill(
                        attention_prompt)));
                check(static_cast<bool>(
                    promotion_session.value()->statistics().vulkan_attention_blocks
                    == 0));

                auto promoted_decode = promotion_session.value()->decode(1);
                auto promotion_cpu_decode = promotion_cpu_session.value()->decode(1);
                check(static_cast<bool>(promoted_decode));
                check(static_cast<bool>(promotion_cpu_decode));
                for (size_t index = 0;
                     index
                     < promotion_cpu_decode.value().logits.size();
                     ++index)
                {
                    check_near(
                        promoted_decode.value().logits[index],
                        promotion_cpu_decode.value().logits[index],
                        1e-4f);
                }
                const SessionStatistics& promoted_statistics = promotion_session.value()->statistics();
                check(static_cast<bool>(
                    promoted_statistics.vulkan_attention_blocks == 0));
                check(static_cast<bool>(
                    promoted_statistics.vulkan_kv_cache_promotions == 0));

                auto continued_decode = promotion_session.value()->decode(0);
                auto continued_cpu_decode = promotion_cpu_session.value()->decode(0);
                check(static_cast<bool>(continued_decode));
                check(static_cast<bool>(continued_cpu_decode));
                for (size_t index = 0;
                     index
                     < continued_cpu_decode.value().logits.size();
                     ++index)
                {
                    check_near(
                        continued_decode.value().logits[index],
                        continued_cpu_decode.value().logits[index],
                        1e-4f);
                }
                check(static_cast<bool>(
                    promotion_session.value()->statistics().vulkan_kv_cache_promotions
                    == 0));
            };
        check_attention_cache_isolation(attention_package.path());

        auto attention_decode = attention_session.value()->decode(1);
        auto cpu_attention_decode = cpu_attention_session.value()->decode(1);
        check(static_cast<bool>(attention_decode));
        check(static_cast<bool>(cpu_attention_decode));
        for (size_t index = 0; index < cpu_attention_decode.value().logits.size(); ++index)
        {
            check_near(attention_decode.value().logits[index], cpu_attention_decode.value().logits[index], 1e-4f);
        }
        for (uint32_t iteration = 0; iteration < 20; ++iteration)
        {
            auto wrapped_decode = attention_session.value()->decode(static_cast<int32_t>(iteration % 2));
            auto wrapped_cpu_decode = cpu_attention_session.value()->decode(static_cast<int32_t>(iteration % 2));
            check(static_cast<bool>(wrapped_decode));
            check(static_cast<bool>(wrapped_cpu_decode));
            for (size_t index = 0; index < wrapped_cpu_decode.value().logits.size(); ++index)
            {
                check_near(wrapped_decode.value().logits[index], wrapped_cpu_decode.value().logits[index], 1e-4f);
            }
        }
        const SessionStatistics& wrapped_statistics = attention_session.value()->statistics();
        check(static_cast<bool>(wrapped_statistics.vulkan_kv_ring_appends == wrapped_statistics.vulkan_attention_blocks));
        check(static_cast<bool>(wrapped_statistics.vulkan_attention_qkv_rope_fusions > 0));
        check(static_cast<bool>(wrapped_statistics.vulkan_attention_qkv_rope_fusions <= wrapped_statistics.vulkan_attention_blocks));
        check(static_cast<bool>(wrapped_statistics.vulkan_attention_qkv_ring_fusions == wrapped_statistics.vulkan_attention_blocks - 1));
        check(static_cast<bool>(wrapped_statistics.vulkan_attention_decode_sdpa_fusions > 0));
        check(static_cast<bool>(wrapped_statistics.vulkan_attention_decode_sdpa_fusions < wrapped_statistics.vulkan_attention_blocks - 1));
        check(static_cast<bool>(wrapped_statistics.vulkan_kv_ring_resizes == 1));
        check(static_cast<bool>(wrapped_statistics.vulkan_kv_ring_wrapped_views > 0));

        Option ncnn_sdpa_ring_options = hybrid_options;
        ncnn_sdpa_ring_options.optimization_flags |= OptimizationVulkanQkvRing;
        ncnn_sdpa_ring_options.optimization_flags &= ~OptimizationVulkanDecodeSdpa;
        auto ncnn_sdpa_ring_model = runtime.load_model(
            attention_package.path(),
            ncnn_sdpa_ring_options);
        check(static_cast<bool>(ncnn_sdpa_ring_model));
        auto ncnn_sdpa_ring_session = runtime.create_session(
            ncnn_sdpa_ring_model.value());
        check(static_cast<bool>(ncnn_sdpa_ring_session));
        auto ncnn_sdpa_ring_prefill = ncnn_sdpa_ring_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(ncnn_sdpa_ring_prefill));
        auto ncnn_sdpa_ring_cpu_session = runtime.create_session(cpu_attention_model.value());
        check(static_cast<bool>(ncnn_sdpa_ring_cpu_session));
        auto ncnn_sdpa_ring_cpu_prefill = ncnn_sdpa_ring_cpu_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(ncnn_sdpa_ring_cpu_prefill));
        auto ncnn_sdpa_ring_decode = ncnn_sdpa_ring_session.value()->decode(1);
        auto ncnn_sdpa_ring_cpu_decode = ncnn_sdpa_ring_cpu_session.value()->decode(1);
        check(static_cast<bool>(ncnn_sdpa_ring_decode));
        check(static_cast<bool>(ncnn_sdpa_ring_cpu_decode));
        for (size_t index = 0; index < ncnn_sdpa_ring_cpu_decode.value().logits.size(); ++index)
        {
            check_near(ncnn_sdpa_ring_decode.value().logits[index], ncnn_sdpa_ring_cpu_decode.value().logits[index], 1e-4f);
        }
        check(static_cast<bool>(ncnn_sdpa_ring_session.value()->statistics().vulkan_attention_qkv_ring_fusions == 1));
        check(static_cast<bool>(ncnn_sdpa_ring_session.value()->statistics().vulkan_attention_decode_sdpa_fusions == 0));

        AttentionPackage full_attention_package(false, 0);
        auto full_attention_model = runtime.load_model(
            full_attention_package.path(),
            attention_options);
        check(static_cast<bool>(full_attention_model));
        SessionOptions chunked_attention_options;
        chunked_attention_options.prefill_chunk_size = 2;
        auto chunked_attention_session = runtime.create_session(full_attention_model.value(), chunked_attention_options);
        check(static_cast<bool>(chunked_attention_session));
        auto chunked_attention_prefill = chunked_attention_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(chunked_attention_prefill));
        check(static_cast<bool>(chunked_attention_session.value()->statistics().kv_cache_allocated_size > chunked_attention_session.value()->statistics().kv_cache_logical_size));

        auto full_cpu_model = runtime.load_model(full_attention_package.path(), cpu_options);
        check(static_cast<bool>(full_cpu_model));
        auto full_cpu_session = runtime.create_session(full_cpu_model.value());
        check(static_cast<bool>(full_cpu_session));
        auto full_cpu_prefill = full_cpu_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(full_cpu_prefill));
        for (size_t index = 0; index < full_cpu_prefill.value().logits.size(); ++index)
        {
            check_near(chunked_attention_prefill.value().logits[index], full_cpu_prefill.value().logits[index], 1e-4f);
        }
        auto chunked_attention_decode = chunked_attention_session.value()->decode(1);
        auto full_cpu_decode = full_cpu_session.value()->decode(1);
        check(static_cast<bool>(chunked_attention_decode));
        check(static_cast<bool>(full_cpu_decode));
        for (size_t index = 0; index < full_cpu_decode.value().logits.size(); ++index)
        {
            check_near(chunked_attention_decode.value().logits[index], full_cpu_decode.value().logits[index], 1e-4f);
        }
        check(static_cast<bool>(chunked_attention_session.value()->statistics().vulkan_attention_qkv_ring_fusions == 1));
        check(static_cast<bool>(chunked_attention_session.value()->statistics().vulkan_attention_decode_sdpa_fusions == 0));

        Option unfused_ring_options = attention_options;
        unfused_ring_options.optimization_flags &= ~OptimizationVulkanQkvRing;
        auto unfused_ring_model = runtime.load_model(
            full_attention_package.path(),
            unfused_ring_options);
        check(static_cast<bool>(unfused_ring_model));
        auto unfused_ring_session = runtime.create_session(
            unfused_ring_model.value());
        check(static_cast<bool>(unfused_ring_session));
        auto unfused_ring_prefill = unfused_ring_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(unfused_ring_prefill));
        auto unfused_ring_decode = unfused_ring_session.value()->decode(1);
        check(static_cast<bool>(unfused_ring_decode));
        check(static_cast<bool>(unfused_ring_session.value()->statistics().vulkan_attention_qkv_ring_fusions == 0));
        check(static_cast<bool>(unfused_ring_session.value()->statistics().vulkan_attention_decode_sdpa_fusions == 0));

        auto prefetch_session = runtime.create_session(hybrid_model.value());
        check(static_cast<bool>(prefetch_session));
        check(static_cast<bool>(prefetch_session.value()->prefill(packed_prompt)));
        check(static_cast<bool>(prefetch_session.value()->statistics().expert_prefetches > 0));
        check(static_cast<bool>(prefetch_session.value()->statistics().expert_prefetch_bytes > 0));
    }
    else
    {
        check(static_cast<bool>(!hybrid_model));
        check(static_cast<bool>(hybrid_model.error().code == ErrorCode::UnsupportedModel));
    }
}

void test_flag_defaults()
{
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationVulkanPipelineBindElision)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationVulkanReadonlyBindings)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationVulkanBatchBufferBarriers)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationVulkanStackDescriptorPayload)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationVulkanCommandGraph)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationCpuLatentVectorSoftmax)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationVulkanLatentInputRmsNorm)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationCpuSimdRmsNorm)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationVulkanRouteAggregation)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationVulkanExpertGpuPriority)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationVulkanIndexedExperts)));
    check(static_cast<bool>(has_flag(
        OptimizationDefaultFlags,
        OptimizationVulkanQnK)));
    check(static_cast<bool>(!has_flag(
        OptimizationDefaultFlags,
        OptimizationCpuPackedWeights)));
    Option runtime;
    check(static_cast<bool>(runtime.optimization_flags == OptimizationDefaultFlags));
    check(static_cast<bool>(runtime.cpu_packed_weight_mode == CpuPackedWeightMode::Disabled));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionMemoryMapExperts)));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionRouterPrediction)));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionForwardAwareCache)));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionRankAdaptivePrefetch)));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionCrossExpertReadCoalescing)));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionAsyncRouterPrediction)));

    RuntimeInfo info;
    check(static_cast<bool>(has_flag(info.flags, RuntimeCpu)));
    check(static_cast<bool>(has_flag(info.flags, RuntimeMxfp4Cpu)));
    check(static_cast<bool>(has_flag(info.flags, RuntimeCrossSession)));

    SchedulerOptions scheduler;
    check(static_cast<bool>(scheduler.num_threads == 0));
    check(static_cast<bool>(scheduler.use_staged_decode));

    ExpertDispatchOptions dispatch;
    check(dispatch.normalization == RouterNormalization::SelectedExperts);

    MoeDescriptor moe;
    check(moe.normalization == RouterNormalization::SelectedExperts);
    check(moe.shared_expert_count == 0);

    LayerDescriptor layer;
    check(layer.ffn.kind == FfnKind::Moe);
    check(layer.attention.kind == AttentionKind::None);

    AttentionBlockPlan attention_plan;
    check(attention_plan.kind == AttentionKind::None);
    CompiledLayerPlan layer_plan;
    check(layer_plan.attention.kind == AttentionKind::None);

    ExpertPlan expert;
    check(static_cast<bool>(expert.layout == ExpertLayout::GateUpDown));

    ModelMemoryPlan memory;
    check(static_cast<bool>(memory.selected_mode == ExpertMemoryMode::Eager));
}

void test_cpu_task_worker()
{
    std::mutex mutex;
    std::condition_variable started_condition;
    std::condition_variable release_condition;
    bool release = false;
    uint32_t started = 0;
    std::atomic<uint32_t> completed = 0;
    auto blocking_task = [&] {
        {
            std::unique_lock<std::mutex> lock(mutex);
            ++started;
            started_condition.notify_all();
            release_condition.wait(lock, [&] {
                return release;
            });
        }
        completed.fetch_add(1, std::memory_order_relaxed);
    };

    bool first_submitted = false;
    bool second_submitted = false;
    bool third_submitted = false;
    {
        CpuTaskWorker worker(2);
        first_submitted = worker.try_submit(blocking_task);
        second_submitted = worker.try_submit(blocking_task);
        third_submitted = worker.try_submit([] {});
        if (first_submitted || second_submitted)
        {
            {
                std::unique_lock<std::mutex> lock(mutex);
                started_condition.wait(lock, [&] {
                    return started != 0;
                });
                release = true;
            }
            release_condition.notify_all();
        }
    }
    check(first_submitted);
    check(second_submitted);
    check(!third_submitted);
    check(completed.load(std::memory_order_relaxed) == 2);

    std::atomic<uint32_t> drained = 0;
    bool drain_submitted = false;
    {
        CpuTaskWorker draining_worker(1);
        drain_submitted = draining_worker.try_submit([&] {
            drained.fetch_add(1, std::memory_order_relaxed);
        });
    }
    check(drain_submitted);
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

void test_float_scale_inplace_and_scaled_add()
{
    std::array<float, 19> values = {};
    std::array<float, 19> output = {};
    std::array<float, 19> expected_values = {};
    std::array<float, 19> expected_output = {};
    constexpr float value_scale = 0.625f;
    constexpr float output_scale = -0.375f;
    for (size_t index = 0; index < values.size(); ++index)
    {
        values[index] = static_cast<float>(static_cast<int>(index % 11) - 5)
                        * 0.125f;
        output[index] = static_cast<float>(static_cast<int>(index % 7) - 3)
                        * 0.25f;
        expected_values[index] = values[index] * value_scale;
        expected_output[index] = output[index]
                                 + output_scale * expected_values[index];
    }
    float_scale_inplace_and_scaled_add(
        values.data(),
        value_scale,
        output.data(),
        output_scale,
        static_cast<uint32_t>(values.size()));
    for (size_t index = 0; index < values.size(); ++index)
    {
        check_near(values[index], expected_values[index], 1e-6f);
        check_near(output[index], expected_output[index], 1e-6f);
    }
}

void test_float_scale_add()
{
    std::array<float, 23> output = {};
    std::array<float, 23> input = {};
    constexpr float output_scale = -0.375f;
    constexpr float input_scale = 0.625f;
    for (size_t index = 0; index < output.size(); ++index)
    {
        output[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.125f;
        input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.0625f;
    }
    std::array<float, 23> expected = output;
    for (size_t index = 0; index < expected.size(); ++index)
        expected[index] = expected[index] * output_scale + input[index] * input_scale;
    float_scale_add(
        output.data(),
        output_scale,
        input.data(),
        input_scale,
        static_cast<uint32_t>(output.size()));
    for (size_t index = 0; index < output.size(); ++index)
        check_near(output[index], expected[index], 1e-6f);
}

void test_float_dot()
{
    std::array<float, 137> left = {};
    std::array<float, 137> right = {};
    float expected = 0.0f;
    for (size_t index = 0; index < left.size(); ++index)
    {
        left[index] = static_cast<float>(static_cast<int>(index % 19) - 9) * 0.125f;
        right[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.0625f;
        expected += left[index] * right[index];
    }
    check_near(
        float_dot(left.data(), right.data(), static_cast<uint32_t>(left.size())),
        expected,
        1e-4f);
    check_near(float_dot(left.data(), right.data(), 0), 0.0f, 1e-6f);
}

void test_float_exp_inplace()
{
    std::array<float, 37> values = {};
    std::array<float, 37> expected = {};
    for (size_t index = 0; index < values.size(); ++index)
    {
        values[index] = -6.0f + static_cast<float>(index) * 0.25f;
        expected[index] = std::exp(values[index]);
    }
    float_exp_inplace(values.data(), static_cast<uint32_t>(values.size()));
    for (size_t index = 0; index < values.size(); ++index)
        check_near(values[index], expected[index], 2e-4f);
    float_exp_inplace(values.data(), 0);
}

void test_float_approximate_exp()
{
    for (int step = -80; step <= 80; ++step)
    {
        const float value = static_cast<float>(step) * 0.25f;
        const float expected = std::exp(value);
        const float tolerance = std::max(1e-6f, std::abs(expected) * 5e-4f);
        check_near(float_approximate_exp(value), expected, tolerance);
    }
    check(std::isinf(float_approximate_exp(104.0f)));
    check(float_approximate_exp(-104.0f) == 0.0f);
    check(std::isnan(float_approximate_exp(std::numeric_limits<float>::quiet_NaN())));
}

void test_int8_float_dot()
{
    std::array<int8_t, 137> left = {};
    std::array<float, 137> right = {};
    float expected = 0.0f;
    for (size_t index = 0; index < left.size(); ++index)
    {
        left[index] = static_cast<int8_t>(static_cast<int>(index % 23) - 11);
        right[index] = static_cast<float>(static_cast<int>(index % 17) - 8)
                       * 0.0625f;
        expected += static_cast<float>(left[index]) * right[index];
    }
    check_near(
        int8_float_dot(left.data(), right.data(), static_cast<uint32_t>(left.size())),
        expected,
        1e-4f);
    check_near(int8_float_dot(left.data(), right.data(), 0), 0.0f, 1e-6f);

#if defined(_MSC_VER) && defined(_M_X64)
    if ((cpu_isa_flags() & CpuIsaX86Avx2Fma) != 0)
    {
        // Exercise AVX2 directly so AVX512-capable hosts cannot mask its tail path.
        check_near(
            msvc_avx2_int8_float_dot(
                left.data(), right.data(), static_cast<uint32_t>(left.size())),
            expected,
            1e-4f);
        constexpr std::array<uint32_t, 23> counts = {
            0,
            1,
            4,
            7,
            8,
            9,
            12,
            15,
            16,
            23,
            24,
            31,
            32,
            33,
            39,
            40,
            47,
            48,
            55,
            56,
            63,
            64,
            137,
        };
        std::array<int8_t, 137> avx2_left = {};
        std::array<float, 137> avx2_right = {};
        for (size_t index = 0; index < avx2_left.size(); ++index)
        {
            avx2_left[index] = static_cast<int8_t>(index % 13 + 1);
            avx2_right[index] = static_cast<float>(index % 7 + 1) * 0.125f;
        }
        for (const uint32_t count : counts)
        {
            float avx2_expected = 0.0f;
            for (uint32_t index = 0; index < count; ++index)
            {
                avx2_expected += static_cast<float>(avx2_left[index])
                                 * avx2_right[index];
            }
            check_near(
                msvc_avx2_int8_float_dot(
                    avx2_left.data(), avx2_right.data(), count),
                avx2_expected,
                1e-4f);
        }
    }
#endif
}

void test_weighted_scale()
{
    std::array<float, 35> input = {};
    std::array<float, 35> float_weight = {};
    std::array<uint16_t, 35> bfloat16_weight = {};
    std::array<float, 35> float_output = {};
    std::array<float, 35> bfloat16_output = {};
    constexpr float scale = 0.3125f;
    constexpr float offset = 0.125f;
    for (size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(static_cast<int>(index % 9) - 4)
                       * 0.1875f;
        float_weight[index] = static_cast<float>(static_cast<int>(index % 7) - 3)
                              * 0.25f;
        bfloat16_weight[index] = float_to_bfloat16(float_weight[index]);
    }
    float_weighted_scale(
        float_output.data(), input.data(), float_weight.data(), scale, offset,
        static_cast<uint32_t>(input.size()));
    bfloat16_weighted_scale(
        bfloat16_output.data(), input.data(), bfloat16_weight.data(), scale,
        offset, static_cast<uint32_t>(input.size()));
    for (size_t index = 0; index < input.size(); ++index)
    {
        check_near(
            float_output[index],
            input[index] * scale * (float_weight[index] + offset),
            1e-6f);
        check_near(
            bfloat16_output[index],
            input[index] * scale
                * (bfloat16_to_float(bfloat16_weight[index]) + offset),
            1e-6f);
    }

    float_weighted_scale(
        input.data(), input.data(), float_weight.data(), scale, offset,
        static_cast<uint32_t>(input.size()));
    for (size_t index = 0; index < input.size(); ++index)
        check_near(input[index], float_output[index], 1e-6f);
}

void test_rms_vector_kernels()
{
    constexpr uint32_t count = 137;
    constexpr float epsilon = 1e-5f;
    constexpr float weight_offset = 0.125f;
    std::array<float, count> input = {};
    std::array<float, count> float_weight = {};
    std::array<uint16_t, count> bfloat16_weight = {};
    std::array<float, count> float_output = {};
    std::array<float, count> bfloat16_output = {};
    for (uint32_t index = 0; index < count; ++index)
    {
        input[index] = static_cast<float>(static_cast<int>(index % 23) - 11) * 0.0625f;
        float_weight[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.03125f;
        bfloat16_weight[index] = float_to_bfloat16(float_weight[index]);
    }
    float square_sum = 0.0f;
    for (float value : input)
        square_sum += value * value;
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    float_rms_norm(
        float_output.data(), input.data(), float_weight.data(), epsilon,
        weight_offset, count);
    bfloat16_rms_norm(
        bfloat16_output.data(), input.data(), bfloat16_weight.data(), epsilon,
        weight_offset, count);
    for (uint32_t index = 0; index < count; ++index)
    {
        check_near(
            float_output[index],
            input[index] * inverse_rms * (float_weight[index] + weight_offset),
            2e-4f);
        check_near(
            bfloat16_output[index],
            input[index] * inverse_rms
                * (bfloat16_to_float(bfloat16_weight[index]) + weight_offset),
            2e-4f);
    }

    std::array<float, count> normalized = input;
    float_rms_scale_inplace(normalized.data(), epsilon, count);
    for (uint32_t index = 0; index < count; ++index)
        check_near(normalized[index], input[index] * inverse_rms, 2e-4f);
    float_rms_scale_inplace(normalized.data(), epsilon, 0);
}

void test_l2_vector_kernel()
{
    constexpr uint32_t count = 137;
    constexpr float epsilon = 1e-6f;
    std::array<float, count> values = {};
    for (uint32_t index = 0; index < count; ++index)
        values[index] = static_cast<float>(static_cast<int>(index % 19) - 9) * 0.125f;
    float square_sum = 0.0f;
    for (float value : values)
        square_sum += value * value;
    const float inverse_norm = 1.0f / std::sqrt(square_sum + epsilon);
    const std::array<float, count> input = values;
    float_l2_scale_inplace(values.data(), epsilon, count);
    for (uint32_t index = 0; index < count; ++index)
        check_near(values[index], input[index] * inverse_norm, 2e-5f);
    float_l2_scale_inplace(values.data(), epsilon, 0);
}

void test_hyper_connection_vector_kernels()
{
    constexpr uint32_t hidden_size = 19;
    std::array<float, hidden_size * 4> input = {};
    std::array<float, hidden_size> reduced = {};
    constexpr std::array<float, 4> pre = {0.25f, -0.5f, 0.75f, 1.25f};
    for (size_t index = 0; index < input.size(); ++index)
        input[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.125f;
    float_hc_pre_4(
        reduced.data(), input.data(), pre[0], pre[1], pre[2], pre[3], hidden_size);
    for (uint32_t index = 0; index < hidden_size; ++index)
    {
        float expected = 0.0f;
        for (uint32_t copy = 0; copy < 4; ++copy)
            expected += input[static_cast<size_t>(copy) * hidden_size + index] * pre[copy];
        check_near(reduced[index], expected, 2e-5f);
    }

    std::array<float, hidden_size> branch = {};
    std::array<float, hidden_size * 4> residual = {};
    std::array<float, 4> post = {0.5f, 1.0f, -0.75f, 1.5f};
    std::array<float, 16> combine = {};
    std::array<float, hidden_size * 4> output = {};
    for (uint32_t index = 0; index < hidden_size; ++index)
        branch[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.25f;
    for (size_t index = 0; index < residual.size(); ++index)
        residual[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.125f;
    for (size_t index = 0; index < combine.size(); ++index)
        combine[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * 0.0625f;
    float_hc_post_4(
        output.data(), branch.data(), residual.data(), post.data(), combine.data(), hidden_size);
    for (uint32_t output_index = 0; output_index < 4; ++output_index)
    {
        for (uint32_t index = 0; index < hidden_size; ++index)
        {
            float expected = branch[index] * post[output_index];
            for (uint32_t residual_index = 0; residual_index < 4; ++residual_index)
                expected += residual[static_cast<size_t>(residual_index) * hidden_size + index]
                            * combine[residual_index * 4 + output_index];
            check_near(
                output[static_cast<size_t>(output_index) * hidden_size + index],
                expected,
                2e-5f);
        }
    }
}

void test_float_rope_kernel()
{
    constexpr uint32_t dimension = 34;
    constexpr uint32_t half_dimension = dimension / 2;
    std::array<float, dimension> values = {};
    std::array<float, half_dimension> cosine = {};
    std::array<float, half_dimension> sine = {};
    std::array<float, dimension> expected = {};
    for (uint32_t index = 0; index < dimension; ++index)
        values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.125f;
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        cosine[index] = 0.75f + static_cast<float>(index % 5) * 0.03125f;
        sine[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.0625f;
    }
    expected = values;
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const float first = values[index];
        const float second = values[half_dimension + index];
        expected[index] = first * cosine[index] - second * sine[index];
        expected[half_dimension + index] = second * cosine[index] + first * sine[index];
    }
    float_rope_inplace(values.data(), cosine.data(), sine.data(), dimension);
    for (uint32_t index = 0; index < dimension; ++index)
        check_near(values[index], expected[index], 2e-5f);
}

void test_float_silu_mul()
{
    std::array<float, 97> gate = {};
    std::array<float, 97> up = {};
    std::array<float, 97> output = {};
    for (size_t index = 0; index < gate.size(); ++index)
    {
        gate[index] = -7.0f + static_cast<float>(index % 29) * 0.5f;
        up[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.125f;
    }
    constexpr float sigmoid_scale = 1.702f;
    constexpr float up_offset = 1.0f;
    float_silu_mul(
        output.data(), gate.data(), up.data(), sigmoid_scale, up_offset,
        static_cast<uint32_t>(output.size()));
    for (size_t index = 0; index < output.size(); ++index)
    {
        const float gate_value = gate[index];
        const float expected = gate_value / (1.0f + std::exp(-sigmoid_scale * gate_value))
                               * (up[index] + up_offset);
        check_near(output[index], expected, 5e-4f);
    }
}

void test_float_sigmoid_mul()
{
    std::array<float, 97> gate = {};
    std::array<float, 97> input = {};
    std::array<float, 97> output = {};
    for (size_t index = 0; index < gate.size(); ++index)
    {
        gate[index] = -7.0f + static_cast<float>(index % 29) * 0.5f;
        input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.125f;
    }
    float_sigmoid_mul(
        output.data(),
        gate.data(),
        input.data(),
        static_cast<uint32_t>(output.size()));
    for (size_t index = 0; index < output.size(); ++index)
        check_near(
            output[index],
            input[index] / (1.0f + std::exp(-gate[index])),
            5e-4f);
}

void test_bfloat16_vector_kernels()
{
    std::array<uint16_t, 129> weights = {};
    std::array<float, 129> input = {};
    for (size_t index = 0; index < weights.size(); ++index)
    {
        const float value = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.375f;
        weights[index] = float_to_bfloat16(value);
        input[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * 0.125f;
    }

    for (uint32_t count : {
             0u, 1u, 7u, 8u, 15u, 16u, 17u, 32u, 33u,
             63u, 64u, 65u, 127u, 128u, 129u})
    {
        float expected_dot = 0.0f;
        float expected_pair_dot = 0.0f;
        std::array<uint16_t, 129> pair_input = {};
        for (uint32_t index = 0; index < count; ++index)
        {
            expected_dot += bfloat16_to_float(weights[index]) * input[index];
            pair_input[index] = float_to_bfloat16(input[index]);
            expected_pair_dot += bfloat16_to_float(weights[index])
                                 * bfloat16_to_float(pair_input[index]);
        }
        check_near(bfloat16_dot(weights.data(), input.data(), count), expected_dot, 1e-4f);
        check_near(
            bfloat16_pair_dot(
                weights.data(),
                pair_input.data(),
                count),
            expected_pair_dot,
            1e-4f);

        std::array<float, 129> output = {};
        std::array<float, 129> expected_output = {};
        constexpr float scale = -0.625f;
        for (uint32_t index = 0; index < count; ++index)
        {
            output[index] = static_cast<float>(static_cast<int>(index % 5) - 2) * 0.25f;
            expected_output[index] = output[index] + scale * bfloat16_to_float(weights[index]);
        }
        bfloat16_scaled_add(output.data(), weights.data(), scale, count);
        for (uint32_t index = 0; index < count; ++index)
            check_near(output[index], expected_output[index], 1e-5f);
    }
}

void test_float_to_bfloat16_array()
{
    std::array<float, 37> input = {};
    std::array<uint16_t, 37> output = {};
    for (size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(static_cast<int>(index % 17) - 8)
                           * 0.03125f
                       + static_cast<float>(index) * 1e-5f;
    }
    float_to_bfloat16_array(
        output.data(),
        input.data(),
        static_cast<uint32_t>(input.size()));
    for (size_t index = 0; index < input.size(); ++index)
        check(static_cast<bool>(output[index] == float_to_bfloat16(input[index])));
}

void test_activation_buffer_reset()
{
    ActivationBuffer buffer(2, 3, DType::Int8);
    std::fill(buffer.mutable_bytes().begin(), buffer.mutable_bytes().end(), std::byte{0x5a});
    bool rejected = false;
    try
    {
        // Exceed vector's size limit without attempting a large allocation.
        buffer.reset(std::vector<std::byte>().max_size() + 1, 1, false);
    }
    catch (const std::length_error&)
    {
        rejected = true;
    }
    check(rejected);
    check(buffer.rows() == 2);
    check(buffer.columns() == 3);
    check(buffer.bytes().size() == 6);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0x5a});

    rejected = false;
    try
    {
        // This product would wrap to zero before vector::resize sees it.
        buffer.reset(std::numeric_limits<size_t>::max() / 2 + 1, 2, false);
    }
    catch (const std::length_error&)
    {
        rejected = true;
    }
    check(rejected);
    check(buffer.rows() == 2);
    check(buffer.columns() == 3);
    check(buffer.bytes().size() == 6);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0x5a});

    CpuBatch typed(1, 2, DType::BFloat16);
    rejected = false;
    try
    {
        typed.reset(std::numeric_limits<size_t>::max() / sizeof(uint16_t) + 1, 1, false);
    }
    catch (const std::length_error&)
    {
        rejected = true;
    }
    check(rejected);
    check(typed.rows() == 1);
    check(typed.columns() == 2);
    check(typed.dtype() == DType::BFloat16);
    check(typed.element_size() == sizeof(uint16_t));
    check(typed.bytes().size() == 2 * sizeof(uint16_t));

    rejected = false;
    try
    {
        ActivationBuffer packed(1, 2, DType::MxFp4);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    check(rejected);

    buffer.reset(1, 3, false);
    check(buffer.rows() == 1);
    check(buffer.columns() == 3);
    check(buffer.bytes().size() == 3);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0x5a});

    buffer.reset(3, 3, true);
    check(buffer.rows() == 3);
    check(buffer.columns() == 3);
    check(buffer.bytes().size() == 9);
    for (std::byte value : buffer.bytes())
        check(value == std::byte{0});
}

void test_bfloat16_batched_linear_kernel()
{
    constexpr size_t token_count = 5;
    constexpr uint32_t input_columns = 512;
    constexpr uint32_t output_columns = 2048;
    std::vector<uint16_t> weights(
        static_cast<size_t>(output_columns) * input_columns);
    std::vector<float> input(token_count * input_columns);
    std::vector<float> output(token_count * output_columns, -7.0f);
    std::vector<float> expected(token_count * output_columns);
    for (size_t index = 0; index < weights.size(); ++index)
    {
        weights[index] = float_to_bfloat16(
            static_cast<float>(static_cast<int>((index * 17 + 5) % 67) - 33)
            * 0.0009765625f);
    }
    for (size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(static_cast<int>((index * 13 + 7) % 29) - 14)
                           * 0.03125f
                       + static_cast<float>(static_cast<int>(index % 11) - 5) * 1e-5f;
    }
    for (size_t token = 0; token < token_count; ++token)
    {
        for (uint32_t output_column = 0;
             output_column < output_columns;
             ++output_column)
        {
            expected[token * output_columns + output_column] = bfloat16_dot(
                weights.data()
                    + static_cast<size_t>(output_column) * input_columns,
                input.data() + token * input_columns,
                input_columns);
        }
    }

    const uint64_t batched_flags = OptimizationCpuBfloat16Batched;
    const uint64_t disabled_flags = g_test_optimization_flags & ~batched_flags;
    const uint64_t enabled_flags = g_test_optimization_flags | batched_flags;
    check(!bfloat16_batched_linear(
        weights.data(),
        input.data(),
        input_columns,
        token_count,
        output_columns,
        input_columns,
        output.data(),
        output_columns,
        4,
        disabled_flags));
    check(static_cast<bool>(
        std::all_of(output.begin(), output.end(), [](float value) {
            return value == -7.0f;
        })));

    bool dispatched = false;
    dispatched = bfloat16_batched_linear(
        weights.data(),
        input.data(),
        input_columns,
        token_count,
        output_columns,
        input_columns,
        output.data(),
        output_columns,
        4,
        enabled_flags);
    if (dispatched)
    {
        for (size_t index = 0; index < output.size(); ++index)
            check_near(output[index], expected[index], 5e-5f);

        std::vector<float> single_output(output_columns, -9.0f);
        check(static_cast<bool>(bfloat16_batched_linear(
            weights.data(),
            input.data(),
            input_columns,
            1,
            output_columns,
            input_columns,
            single_output.data(),
            output_columns,
            4,
            enabled_flags)));
        for (uint32_t output_column = 0;
             output_column < output_columns;
             ++output_column)
        {
            check(static_cast<bool>(
                single_output[output_column] == expected[output_column]));
        }

        Bfloat16BatchedLinearExecutionCounter first_counter;
        Bfloat16BatchedLinearExecutionCounter second_counter;
        std::vector<float> first_output(output.size());
        std::vector<float> second_output(output.size());
        std::atomic<uint32_t> ready{0};
        std::atomic<bool> start{false};
        std::atomic<bool> scoped_dispatches_succeeded{true};
        auto run_scoped = [&](Bfloat16BatchedLinearExecutionCounter& counter,
                              std::vector<float>& scoped_output) {
            const ScopedBfloat16BatchedLinearExecutionCounter scope(&counter);
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            if (!bfloat16_batched_linear(
                    weights.data(),
                    input.data(),
                    input_columns,
                    token_count,
                    output_columns,
                    input_columns,
                    scoped_output.data(),
                    output_columns,
                    2,
                    enabled_flags))
            {
                scoped_dispatches_succeeded.store(
                    false,
                    std::memory_order_relaxed);
            }
        };
        std::thread first_worker(
            run_scoped,
            std::ref(first_counter),
            std::ref(first_output));
        std::thread second_worker(
            run_scoped,
            std::ref(second_counter),
            std::ref(second_output));
        while (ready.load(std::memory_order_relaxed) != 2)
            std::this_thread::yield();
        start.store(true, std::memory_order_release);
        first_worker.join();
        second_worker.join();
        check(scoped_dispatches_succeeded.load(std::memory_order_relaxed));
        check(static_cast<bool>(first_counter.dispatch_count() == 1));
        check(static_cast<bool>(second_counter.dispatch_count() == 1));
    }
    else
    {
        check(std::string(bfloat16_batched_linear_kernel_name(enabled_flags)) == "unavailable");
    }
}

void benchmark_bfloat16_attention_kernels()
{
    constexpr uint32_t head_dimension = 128;
    constexpr uint32_t token_count = 4096;
    constexpr uint32_t iterations = 16;
    constexpr float probability = 0.03125f;
    std::vector<uint16_t> keys(static_cast<size_t>(token_count) * head_dimension);
    std::vector<uint16_t> values(static_cast<size_t>(token_count) * head_dimension);
    std::vector<float> key_float(keys.size());
    std::vector<float> value_float(values.size());
    std::array<float, head_dimension> query = {};
    std::array<float, head_dimension> output = {};
    for (uint32_t column = 0; column < head_dimension; ++column)
        query[column] = static_cast<float>(static_cast<int>(column % 17) - 8) * 0.03125f;
    for (size_t index = 0; index < keys.size(); ++index)
    {
        const float key = static_cast<float>(static_cast<int>(index % 31) - 15) * 0.015625f;
        const float value = static_cast<float>(static_cast<int>(index % 23) - 11) * 0.0234375f;
        keys[index] = float_to_bfloat16(key);
        values[index] = float_to_bfloat16(value);
    }

    float baseline_checksum = 0.0f;
    float direct_checksum = 0.0f;
    const auto baseline_start = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        for (size_t index = 0; index < keys.size(); ++index)
        {
            key_float[index] = bfloat16_to_float(keys[index]);
            value_float[index] = bfloat16_to_float(values[index]);
        }
        output.fill(0.0f);
        for (uint32_t token = 0; token < token_count; ++token)
        {
            const size_t offset = static_cast<size_t>(token) * head_dimension;
            baseline_checksum += float_dot(query.data(), key_float.data() + offset, head_dimension);
            float_scaled_add(output.data(), value_float.data() + offset, probability, head_dimension);
        }
    }
    const auto baseline_end = std::chrono::steady_clock::now();

    const auto direct_start = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        output.fill(0.0f);
        for (uint32_t token = 0; token < token_count; ++token)
        {
            const size_t offset = static_cast<size_t>(token) * head_dimension;
            direct_checksum += bfloat16_dot(keys.data() + offset, query.data(), head_dimension);
            bfloat16_scaled_add(output.data(), values.data() + offset, probability, head_dimension);
        }
    }
    const auto direct_end = std::chrono::steady_clock::now();

    const double baseline_ms = std::chrono::duration<double, std::milli>(baseline_end - baseline_start).count();
    const double direct_ms = std::chrono::duration<double, std::milli>(direct_end - direct_start).count();
    std::cout << std::fixed << std::setprecision(3)
              << "bfloat16_attention_ab baseline_ms=" << baseline_ms
              << " direct_ms=" << direct_ms
              << " speedup=" << (baseline_ms / direct_ms)
              << " checksum_delta=" << std::abs(baseline_checksum - direct_checksum) << '\n';
}

void benchmark_qnk_gemm()
{
    constexpr uint32_t columns = 4096;
    constexpr size_t rows = 32;
    constexpr size_t token_count = 8;
    constexpr uint32_t iterations = 24;
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };

    CpuBatch input(token_count, columns);
    for (size_t token = 0; token < token_count; ++token)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            input.row(token)[column] = static_cast<float>(static_cast<int>((column + token * 3) % 31) - 15)
                                       * 0.03125f;
        }
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "qnk_gemm_kernel=" << mxfp4_kernel_name() << '\n';
    for (const DType dtype : dtypes)
    {
        const size_t raw_bytes = static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns));
        std::vector<uint8_t> raw(raw_bytes, 0);
        const size_t row_bytes = raw_bytes / rows;
        const size_t block_bytes = qnk_block_bytes(dtype);
        const uint32_t block_count = columns / qnk_block_elements;
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + row * row_bytes + static_cast<size_t>(block) * block_bytes;
                if (dtype == DType::Q2K)
                {
                    std::fill_n(encoded, 16, uint8_t{1});
                    encoded[80] = 0x00;
                    encoded[81] = 0x3c;
                    encoded[82] = 0;
                    encoded[83] = 0;
                    for (uint32_t index = 0; index < 64; ++index)
                        encoded[16 + index] = static_cast<uint8_t>(index * 7u + row + block);
                }
                else if (dtype == DType::Q3K)
                {
                    std::fill_n(encoded + 96, 12, uint8_t{0});
                    encoded[108] = 0x00;
                    encoded[109] = 0x3c;
                    for (uint32_t index = 0; index < 32; ++index)
                        encoded[index] = static_cast<uint8_t>(index * 5u + row);
                    for (uint32_t index = 0; index < 64; ++index)
                        encoded[32 + index] = static_cast<uint8_t>(index * 11u + row + block);
                }
                else if (dtype == DType::Q4K || dtype == DType::Q5K)
                {
                    encoded[0] = 0x00;
                    encoded[1] = 0x3c;
                    encoded[2] = 0;
                    encoded[3] = 0;
                    std::fill_n(encoded + 4, 8, uint8_t{1});
                    for (uint32_t index = 0; index < 128; ++index)
                        encoded[(dtype == DType::Q4K ? 16u : 48u) + index] = static_cast<uint8_t>(index * 13u + row + block);
                }
                else if (dtype == DType::Q6K)
                {
                    std::fill_n(encoded + 192, 16, uint8_t{1});
                    encoded[208] = 0x00;
                    encoded[209] = 0x3c;
                    for (uint32_t index = 0; index < 128; ++index)
                        encoded[index] = static_cast<uint8_t>(index * 7u + row + block);
                    for (uint32_t index = 0; index < 64; ++index)
                        encoded[128 + index] = static_cast<uint8_t>(index * 5u + row);
                }
                else
                {
                    const float scale = 1.0f;
                    std::memcpy(encoded, &scale, sizeof(scale));
                    for (uint32_t index = 0; index < qnk_block_elements; ++index)
                        encoded[4 + index] = static_cast<uint8_t>(index * 9u + row + block);
                }
            }
        }

        TensorData matrix;
        matrix.dtype = dtype;
        matrix.shape = {static_cast<uint32_t>(rows), columns};
        matrix.quantized_data = std::move(raw);
        CpuBatch direct_output;
        linear_batch_into(matrix, input, direct_output, 0);
        for (uint32_t iteration = 0; iteration < 2; ++iteration)
            linear_batch_into(matrix, input, direct_output, 0);

        const std::span<const uint8_t> raw_view = matrix.qnk_values();
        float reference_checksum = 0.0f;
        std::vector<float> reference_output(token_count * rows, 0.0f);
        alignas(64) float decoded[qnk_block_elements];
        const auto baseline_start = std::chrono::steady_clock::now();
        for (uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            for (size_t row = 0; row < rows; ++row)
            {
                for (size_t token = 0; token < token_count; ++token)
                {
                    float sum = 0.0f;
                    for (uint32_t block = 0; block < block_count; ++block)
                    {
                        const uint8_t* encoded = raw_view.data()
                                                 + (row * static_cast<size_t>(block_count) + block) * block_bytes;
                        qnk_dequantize_block(dtype, encoded, decoded);
                        sum += float_dot(
                            decoded,
                            input.row(token) + static_cast<size_t>(block) * qnk_block_elements,
                            qnk_block_elements);
                    }
                    reference_output[token * rows + row] = sum;
                }
            }
            reference_checksum += reference_output[(iteration % token_count) * rows + iteration % rows];
        }
        const auto baseline_end = std::chrono::steady_clock::now();

        float direct_checksum = 0.0f;
        const auto direct_start = std::chrono::steady_clock::now();
        for (uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            linear_batch_into(matrix, input, direct_output, 0);
            direct_checksum += direct_output.row(iteration % token_count)[iteration % rows];
        }
        const auto direct_end = std::chrono::steady_clock::now();
        const double baseline_ms = std::chrono::duration<double, std::milli>(baseline_end - baseline_start).count();
        const double direct_ms = std::chrono::duration<double, std::milli>(direct_end - direct_start).count();
        std::cout << "qnk_gemm dtype=" << static_cast<int>(dtype)
                  << " rows=" << rows
                  << " columns=" << columns
                  << " tokens=" << token_count
                  << " iterations=" << iterations
                  << " baseline_ms=" << baseline_ms
                  << " direct_ms=" << direct_ms
                  << " ms=" << direct_ms
                  << " per_iter_ms=" << (direct_ms / iterations)
                  << " speedup=" << (baseline_ms / direct_ms)
                  << " checksum_delta=" << std::abs(reference_checksum - direct_checksum)
                  << " checksum=" << direct_checksum << '\n';
    }
}

void benchmark_vulkan_qnk()
{
    constexpr uint32_t columns = 4096;
    constexpr size_t rows = 256;
    constexpr size_t token_count = 64;
    constexpr uint32_t iterations = 8;
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
    if (get_gpu_count() == 0)
    {
        std::cout << "vulkan_qnk unavailable\n";
        return;
    }

    CpuBatch input(token_count, columns);
    for (size_t token = 0; token < input.rows(); ++token)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(token)[column] = static_cast<float>(static_cast<int>((column * 7 + token * 13) % 29) - 14) * 0.03125f;
    }

    std::cout << std::fixed << std::setprecision(3);
    for (const DType dtype : dtypes)
    {
        const size_t raw_bytes = static_cast<size_t>(qnk_storage_bytes(dtype, rows, columns));
        const size_t row_bytes = raw_bytes / rows;
        const size_t block_bytes = qnk_block_bytes(dtype);
        const uint32_t block_count = columns / qnk_block_elements;
        std::vector<uint8_t> raw(raw_bytes);
        for (size_t index = 0; index < raw.size(); ++index)
            raw[index] = static_cast<uint8_t>((index * 37u + static_cast<uint32_t>(dtype) * 11u + 5u) & 0xffu);
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + row * row_bytes + static_cast<size_t>(block) * block_bytes;
                if (dtype == DType::Q2K)
                {
                    std::fill_n(encoded, 16, uint8_t{1});
                    encoded[80] = 0x00;
                    encoded[81] = 0x3c;
                    encoded[82] = 0;
                    encoded[83] = 0;
                }
                else if (dtype == DType::Q3K)
                {
                    std::fill_n(encoded + 96, 12, uint8_t{0});
                    encoded[108] = 0x00;
                    encoded[109] = 0x3c;
                }
                else if (dtype == DType::Q4K || dtype == DType::Q5K)
                {
                    encoded[0] = 0x00;
                    encoded[1] = 0x3c;
                    encoded[2] = 0;
                    encoded[3] = 0;
                    std::fill_n(encoded + 4, 8, uint8_t{1});
                }
                else if (dtype == DType::Q6K)
                {
                    std::fill_n(encoded + 192, 16, uint8_t{1});
                    encoded[208] = 0x00;
                    encoded[209] = 0x3c;
                }
                else
                {
                    const float scale = 1.0f;
                    std::memcpy(encoded, &scale, sizeof(scale));
                }
            }
        }

        TensorData matrix;
        matrix.dtype = dtype;
        matrix.shape = {static_cast<uint32_t>(rows), columns};
        matrix.quantized_data = std::move(raw);
        CpuBatch cpu_output;
        linear_batch_into(matrix, input, cpu_output, 0);
        auto vulkan = NcnnVulkanQnkOperator::create(
            matrix,
            nullptr,
            automatic_vulkan_device_index,
            context_instance,
            g_test_optimization_flags | OptimizationVulkanQnK);
        if (!vulkan)
        {
            std::cout << "vulkan_qnk dtype=" << static_cast<int>(dtype) << " create=failed\n";
            continue;
        }
        CpuBatch gpu_output;
        for (uint32_t iteration = 0; iteration < 2; ++iteration)
        {
            linear_batch_into(matrix, input, cpu_output, 0);
            if (!vulkan->forward(input, gpu_output))
            {
                std::cout << "vulkan_qnk dtype=" << static_cast<int>(dtype) << " forward=failed\n";
                gpu_output.clear();
                break;
            }
        }
        if (gpu_output.rows() == 0)
            continue;

        const auto cpu_start = std::chrono::steady_clock::now();
        float cpu_checksum = 0.0f;
        for (uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            linear_batch_into(matrix, input, cpu_output, 0);
            cpu_checksum += cpu_output.row(iteration % token_count)[iteration % rows];
        }
        const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_start).count();

        const auto gpu_start = std::chrono::steady_clock::now();
        float gpu_checksum = 0.0f;
        for (uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            if (!vulkan->forward(input, gpu_output))
            {
                gpu_output.clear();
                break;
            }
            gpu_checksum += gpu_output.row(iteration % token_count)[iteration % rows];
        }
        const double gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu_start).count();
        if (gpu_output.rows() == 0)
        {
            std::cout << "vulkan_qnk dtype=" << static_cast<int>(dtype) << " forward=failed\n";
            continue;
        }
        float max_error = 0.0f;
        for (size_t token = 0; token < token_count; ++token)
        {
            for (size_t row = 0; row < rows; ++row)
                max_error = std::max(max_error, std::fabs(gpu_output.row(token)[row] - cpu_output.row(token)[row]));
        }
        std::cout << "vulkan_qnk dtype=" << static_cast<int>(dtype)
                  << " rows=" << rows
                  << " columns=" << columns
                  << " tokens=" << token_count
                  << " iterations=" << iterations
                  << " cpu_ms=" << cpu_ms
                  << " gpu_ms=" << gpu_ms
                  << " speedup=" << (cpu_ms / gpu_ms)
                  << " max_error=" << max_error
                  << " checksum_delta=" << std::fabs(cpu_checksum - gpu_checksum)
                  << '\n';
    }
}

void benchmark_vulkan_qnk_expert()
{
    constexpr uint32_t hidden_columns = 512;
    constexpr uint32_t intermediate_columns = 256;
    constexpr size_t token_count = 64;
    constexpr uint32_t iterations = 8;
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
    if (get_gpu_count() == 0)
    {
        std::cout << "vulkan_qnk_expert unavailable\n";
        return;
    }

    const auto make_q4k = [](size_t rows, uint32_t columns, uint32_t seed) {
        std::vector<uint8_t> raw(static_cast<size_t>(qnk_storage_bytes(DType::Q4K, rows, columns)));
        const size_t block_bytes = qnk_block_bytes(DType::Q4K);
        const uint32_t block_count = columns / qnk_block_elements;
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t block = 0; block < block_count; ++block)
            {
                uint8_t* encoded = raw.data() + (row * block_count + block) * block_bytes;
                encoded[0] = 0x00;
                encoded[1] = 0x3c;
                encoded[2] = 0x00;
                encoded[3] = 0x3c;
                std::fill_n(encoded + 4, 8, uint8_t{1});
                for (uint32_t index = 0; index < 64; ++index)
                    encoded[16 + index] = static_cast<uint8_t>(0x11u + ((seed + row + block + index) & 0x0fu));
            }
        }
        return raw;
    };

    TensorData gate_up;
    gate_up.dtype = DType::Q4K;
    gate_up.shape = {intermediate_columns * 2, hidden_columns};
    gate_up.quantized_data = make_q4k(gate_up.shape[0], hidden_columns, 7);
    TensorData down;
    down.dtype = DType::Q4K;
    down.shape = {hidden_columns, intermediate_columns};
    down.quantized_data = make_q4k(down.shape[0], intermediate_columns, 19);

    CpuBatch input(token_count, hidden_columns);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((row * 17 + column * 5) % 31) - 15) * 0.015625f;
    }

    auto vulkan = NcnnVulkanQnkExpertOperator::create(
        gate_up,
        nullptr,
        down,
        nullptr,
        0.0f,
        automatic_vulkan_device_index,
        ExpertActivation::GptOssSwiGlu,
        context_instance,
        g_test_optimization_flags | OptimizationVulkanQnK);
    if (!vulkan)
    {
        std::cout << "vulkan_qnk_expert create=failed\n";
        return;
    }

    const auto cpu_forward = [&]() {
        const CpuBatch gate_up_output = linear_batch(gate_up, input, 0);
        CpuBatch activated(token_count, intermediate_columns);
        for (size_t row = 0; row < token_count; ++row)
        {
            for (uint32_t column = 0; column < intermediate_columns; ++column)
            {
                const float gate = gate_up_output.row(row)[column * 2];
                const float up = gate_up_output.row(row)[column * 2 + 1];
                const float silu = gate / (1.0f + std::exp(-1.702f * gate));
                activated.row(row)[column] = silu * (up + 1.0f);
            }
        }
        return linear_batch(down, activated, 0);
    };

    CpuBatch cpu_output = cpu_forward();
    CpuBatch gpu_output;
    for (uint32_t iteration = 0; iteration < 2; ++iteration)
    {
        cpu_output = cpu_forward();
        check(static_cast<bool>(vulkan->forward(input, gpu_output)));
    }

    const auto cpu_start = std::chrono::steady_clock::now();
    float cpu_checksum = 0.0f;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        cpu_output = cpu_forward();
        cpu_checksum += cpu_output.row(iteration % token_count)[iteration % hidden_columns];
    }
    const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_start).count();

    const auto gpu_start = std::chrono::steady_clock::now();
    float gpu_checksum = 0.0f;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        check(static_cast<bool>(vulkan->forward(input, gpu_output)));
        gpu_checksum += gpu_output.row(iteration % token_count)[iteration % hidden_columns];
    }
    const double gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu_start).count();
    float max_error = 0.0f;
    for (size_t row = 0; row < token_count; ++row)
    {
        for (uint32_t column = 0; column < hidden_columns; ++column)
            max_error = std::max(max_error, std::fabs(gpu_output.row(row)[column] - cpu_output.row(row)[column]));
    }
    std::cout << std::fixed << std::setprecision(3)
              << "vulkan_qnk_expert rows=" << hidden_columns
              << " intermediate=" << intermediate_columns
              << " tokens=" << token_count
              << " iterations=" << iterations
              << " cpu_ms=" << cpu_ms
              << " gpu_ms=" << gpu_ms
              << " speedup=" << (cpu_ms / gpu_ms)
              << " max_error=" << max_error
              << " checksum_delta=" << std::fabs(cpu_checksum - gpu_checksum)
              << '\n';
}

void test_cpu_resource_coordination()
{
    const CpuThreadBudget budget = resolve_cpu_thread_budget(1);
    check(budget.num_threads >= 1);
    check(budget.num_threads <= budget.max_threads);
    check(budget.num_io_threads == 1);
    check(choose_cpu_team_size(0, 8, 4, budget.num_threads) == 1);
    check(choose_cpu_team_size(1024, 8, 4, budget.num_threads) <= budget.num_threads);
    check(choose_cpu_team_size(std::numeric_limits<uint64_t>::max(), 2, 4, 8) == 8);

    // Test permit accounting independently of the host topology.
    CpuThreadBudgetController controller({2, 1, 4});
    check(controller.try_acquire_compute(0).empty());
    auto base = controller.acquire_compute(2, false);
    check(base.size() == 2);
    check(controller.try_acquire_compute(1, false).empty());
    auto extra = controller.try_acquire_compute(8);
    check(extra.size() == 2);
    check(controller.available() == 0);
    check(controller.try_acquire_compute(1).empty());

    // Extra permits must not prevent reuse of released base permits.
    base = {};
    check(controller.available(false) == 2);
    auto next = controller.try_acquire_compute(2, false);
    check(next.size() == 2);
    auto moved = std::move(extra);
    check(extra.empty());
    check(moved.size() == 2);
    check(controller.available() == 0);
    next = std::move(moved);
    check(moved.empty());
    check(controller.available(false) == 2);
    next = {};
    check(controller.available() == 4);
    check(controller.available(false) == 2);
    {
        auto scoped = controller.try_acquire_compute(4);
        check(scoped.size() == 4);
    }
    check(controller.available() == 4);

    const CpuThreadBudget scheduler_budget = resolve_cpu_thread_budget();
    constexpr std::array<uint32_t, 5> scheduler_thread_counts = {0, 1, 2, 4, std::numeric_limits<uint32_t>::max()};
    TestRuntime runtime;
    for (uint32_t requested_threads : scheduler_thread_counts)
    {
        SchedulerOptions options;
        options.num_threads = requested_threads;
        auto scheduler = runtime.create_scheduler(options);
        check(static_cast<bool>(scheduler));
        const SchedulerStatistics statistics = scheduler.value()->statistics();
        const uint32_t expected_threads = std::min(requested_threads == 0 ? 4u : requested_threads, scheduler_budget.num_threads);
        check(statistics.num_threads == expected_threads);
    }
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    try
    {
        if (argc > 2)
        {
            std::cerr << "expected at most one test argument\n";
            return 2;
        }
        const std::string argument = argc > 1 ? std::string(argv[1]) : std::string{};
        if (argument == "--benchmark-bfloat16-attention")
        {
            ncnn::moe::benchmark_bfloat16_attention_kernels();
            return 0;
        }
        if (argument == "--benchmark-qnk")
        {
            ncnn::moe::benchmark_qnk_gemm();
            return 0;
        }
        if (argument == "--benchmark-vulkan-qnk")
        {
            ncnn::moe::benchmark_vulkan_qnk();
            return 0;
        }
        if (argument == "--benchmark-vulkan-qnk-expert")
        {
            ncnn::moe::benchmark_vulkan_qnk_expert();
            return 0;
        }
        if (argument == "--disable-vulkan-pipeline-bind-elision")
        {
            ncnn::moe::g_test_optimization_flags &= ~ncnn::moe::OptimizationVulkanPipelineBindElision;
        }
        else if (!argument.empty())
        {
            std::cerr << "unknown test argument: " << argument << '\n';
            return 2;
        }
        ncnn::moe::test_flag_defaults();
        ncnn::moe::test_cpu_task_worker();
        ncnn::moe::test_cpu_resource_coordination();
        ncnn::moe::test_float_scaled_add();
        ncnn::moe::test_float_scale_inplace_and_scaled_add();
        ncnn::moe::test_float_scale_add();
        ncnn::moe::test_float_dot();
        ncnn::moe::test_float_exp_inplace();
        ncnn::moe::test_float_approximate_exp();
        ncnn::moe::test_int8_float_dot();
        ncnn::moe::test_weighted_scale();
        ncnn::moe::test_rms_vector_kernels();
        ncnn::moe::test_l2_vector_kernel();
        ncnn::moe::test_hyper_connection_vector_kernels();
        ncnn::moe::test_float_rope_kernel();
        ncnn::moe::test_float_silu_mul();
        ncnn::moe::test_float_sigmoid_mul();
        ncnn::moe::test_bfloat16_vector_kernels();
        ncnn::moe::test_float_to_bfloat16_array();
        ncnn::moe::test_activation_buffer_reset();
        ncnn::moe::test_bfloat16_batched_linear_kernel();
        ncnn::moe::test_ncnn_linear_operator();
        ncnn::moe::test_dense_mxn_tiles();
        ncnn::moe::test_released_dense_host_storage_guard();
        ncnn::moe::test_ncnn_vulkan_bfloat16_operator();
        ncnn::moe::test_ncnn_vulkan_float8_operator();
        ncnn::moe::test_mxfp4_cpu_kernel_and_fused_gate_up();
        ncnn::moe::test_qnk_cpu_kernel();
        ncnn::moe::test_qnk_graph_gate_up_fusion();
        ncnn::moe::test_ncnn_vulkan_qnk_operator();
        ncnn::moe::test_ncnn_vulkan_qnk_expert_operator();
        ncnn::moe::test_sharded_expert_victim_cache();
        ncnn::moe::test_mapped_file_range_and_shared_buffer();
        ncnn::moe::test_safetensors_dense_mmap();
        ncnn::moe::test_safetensors_packed_mxfp4_expert();
        ncnn::moe::test_safetensors_qnk_source_binding();
        ncnn::moe::test_file_backed_bfloat16_expert_cache();
        ncnn::moe::test_file_backed_mxfp4_expert_cache();
        ncnn::moe::test_cross_session_batch_scheduler();
        ncnn::moe::test_batch_scheduler_submission_contract();
        ncnn::moe::test_independent_batch_scheduler();
        ncnn::moe::test_staged_bfloat16_dispatch_telemetry();
        ncnn::moe::test_chunked_prefill_statistics_commit_on_success();
        ncnn::moe::test_prefill_decode_and_reset();
        ncnn::moe::test_invalid_token_is_transactional();
        ncnn::moe::test_chunked_prefill_matches_single_batch();
        ncnn::moe::test_topk_selected_weight_normalization_and_combine();
        ncnn::moe::test_int8_expert_linear();
        ncnn::moe::test_invalid_int8_scale_is_rejected();
        ncnn::moe::test_attention_kv_cache_and_reset();
        ncnn::moe::test_bfloat16_ring_kv_cache();
        ncnn::moe::test_attention_graph_without_bias_or_sink();
        ncnn::moe::test_shared_expert_descriptor();
        ncnn::moe::test_execution_graph_and_scheduler();
        ncnn::moe::test_expert_dispatcher_groups_routes();
        ncnn::moe::test_deepseek_router_and_hyper_connection_kernels();
        ncnn::moe::test_deepseek_v4_descriptors();
        ncnn::moe::test_qwen3_5_moe_descriptors();
        ncnn::moe::test_qwen4_exp_descriptors();
        ncnn::moe::test_qwen4_exp_compile_and_execute();
        ncnn::moe::test_gated_residual_kernels();
        ncnn::moe::test_ple_prefill_decode_continuation();
        ncnn::moe::test_qsa_prefill_decode_continuation();
        ncnn::moe::test_gated_delta_net_continuation();
        ncnn::moe::test_automatic_expert_memory_planning();
        ncnn::moe::test_sampling_and_streaming_generation();
        ncnn::moe::test_generation_callback_errors();
        ncnn::moe::test_manifest_readers();
        ncnn::moe::test_model_adapter_scopes();
        ncnn::moe::test_loader_reports_adapter_and_weight_errors();
        ncnn::moe::test_backend_capabilities_and_hybrid_execution();
        std::cout << "all ncnn_moe tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
