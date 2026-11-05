#ifndef NCNN_MOE_TEST_RUNTIME_SUPPORT_H
#define NCNN_MOE_TEST_RUNTIME_SUPPORT_H

#include "ncnn/moe/runtime.h"

#include "graph/graph.h"
#include "graph/compiler.h"
#include "kernels/attention.h"
#include "kernels/activationbuffer.h"
#include "kernels/bfloat16.h"
#include "kernels/fastmath.h"
#include "kernels/gateddeltanet.h"
#include "kernels/gatedresidual.h"
#include "kernels/latentattention.h"
#include "kernels/mxfp4.h"
#include "kernels/ple.h"
#include "kernels/qnk.h"
#include "kernels/ops.h"
#include "kernels/statecache.h"
#include "kernels/float8.h"
#include "kernels/int4.h"
#include "kernels/vector.h"
#include "kernels/hyperconnection.h"
#include "engine/executor.h"
#include "engine/cpu.h"
#include "engine/expert.h"
#include "engine/expertbackend.h"
#include "graph/compiledmodel.h"
#include "engine/sessionstate.h"
#include "storage/expertcache.h"
#include "storage/expertresidency.h"
#include "modeladapter_fixture.h"
#include "storage/mappedfile.h"
#include "graph/memoryplan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/mxfp4_weightpacking.h"
#include "backends/ncnn/modelpipeline.h"
#include "backends/ncnn/expertbackend_vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "graph/router.h"
#include "models/modeladapter_gptoss.h"
#include "models/modeladapter.h"
#include "models/json.h"
#include "models/modeladapter_deepseekv4.h"
#include "models/modeladapter_qwen3_5.h"
#include "models/modeladapter_qwen4exp.h"
#include "models/safetensors.h"

#if defined(NCNN_MOE_WITH_NCNN) && NCNN_MOE_WITH_NCNN \
    && defined(_OPENMP)                               \
    && (defined(__clang__) || defined(_OPENMP_LLVM_RUNTIME))
#include <cpu.h>
#endif

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include "kernels/vector_x86.h"
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <Accelerate/Accelerate.h>
#include <sys/sysctl.h>
#endif

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cfenv>
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

static void check_near(float actual,
                       float expected,
                       float tolerance,
                       const std::source_location location = std::source_location::current())
{
    if (!(std::abs(actual - expected) <= tolerance))
    {
        throw std::runtime_error("near check failed at " + std::string(location.file_name()) + ":"
                                 + std::to_string(location.line()) + ": actual="
                                 + std::to_string(actual) + ", expected=" + std::to_string(expected));
    }
}

static std::vector<ExpertStatus> submit_backend_batch(ExpertBackend& backend,
                                                      std::span<const ExpertRequest> requests)
{
    auto submission = backend.submit_batch(requests);
    if (!submission)
        return std::vector<ExpertStatus>(requests.size(), ExpertStatus::Failed);

    const std::span<const ExpertStatus> reservation_span = submission->reservations();
    if (reservation_span.size() != requests.size())
    {
        submission->abort();
        return std::vector<ExpertStatus>(requests.size(), ExpertStatus::Failed);
    }
    const std::vector<ExpertStatus> planned(reservation_span.begin(), reservation_span.end());

    std::vector<ExpertStatus> results = submission->wait();
    if (results.size() != requests.size())
    {
        submission->abort();
        return std::vector<ExpertStatus>(requests.size(), ExpertStatus::Failed);
    }
    for (size_t index = 0; index < results.size(); ++index)
    {
        if (results[index] == ExpertStatus::Executed
            && planned[index] != ExpertStatus::Executed)
        {
            submission->abort();
            return std::vector<ExpertStatus>(requests.size(), ExpertStatus::Failed);
        }
    }

    bool has_executed = false;
    for (ExpertStatus result : results)
        has_executed = has_executed || result == ExpertStatus::Executed;
    if (!has_executed)
    {
        submission->abort();
        return results;
    }

    if (!submission->commit())
    {
        for (ExpertStatus& result : results)
        {
            if (result == ExpertStatus::Executed)
                result = ExpertStatus::Failed;
        }
        submission->abort();
    }
    return results;
}

static float bfloat16_storage_tolerance(float value) noexcept
{
    // Native conversion may round, while ncnn's packed fallback truncates.
    // One storage ULP is the strict bound shared by both paths.
    if (value == 0.0f)
        return std::ldexp(1.0f, -133);
    return std::ldexp(1.0f,
                      std::max(std::ilogb(std::abs(value)) - 7, -133));
}

inline uint64_t g_test_optimization_flags = OptimizationDefaultFlags;

static bool has_usable_vulkan_device(const RuntimeInfo& info, uint32_t capability_flag) noexcept
{
    return has_flag(info.flags, capability_flag)
           && info.default_gpu_index < info.gpu_infos.size()
           && info.gpu_infos[info.default_gpu_index].type != VulkanDeviceType::Cpu;
}

class TestRuntime final : public Runtime
{
public:
    TestRuntime()
    {
        register_adapter(std::make_shared<FixtureModelAdapter>());
    }
};

void test_batch_combine_backend();
void test_dense_ffn();
void test_int4();
void test_quant_archive();
void test_native_awq();

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
    explicit TemporaryModelPackage(bool use_token_experts = false)
    {
        path_ = create_unique_test_directory("ncnn_moe_phase0_test_");
        write_valid_manifest(use_token_experts);
        write_valid_weights(use_token_experts);
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
    void write_valid_manifest(bool use_token_experts)
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
  "norm_epsilon": 0.00001,)";
        if (use_token_experts)
            manifest << R"(
  "hash_routing_layer_count": 1,)";
        manifest << R"(
  "weights_file": "model.test.bin"
})";
    }

    void write_valid_weights(bool use_token_experts)
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
            use_token_experts ? 3.0f : 2.0f,
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
        if (use_token_experts)
        {
            const std::array<int64_t, 4> token_experts = {1, 0, 1, 0};
            weights.write(reinterpret_cast<const char*>(token_experts.data()),
                          static_cast<std::streamsize>(token_experts.size() * sizeof(int64_t)));
        }
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
    static void write_floats(std::ofstream& stream,
                             const std::vector<float>& values)
    {
        stream.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    static void write_bfloat16(std::ofstream& stream,
                               const std::vector<uint16_t>& values)
    {
        stream.write(reinterpret_cast<const char*>(values.data()),
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
    explicit AttentionPackage(bool bfloat16_kv_cache = false, uint32_t sliding_window = 2, bool attention_bias = true,
                              bool attention_sinks = true, bool grouped_query = false, uint32_t context_length = 0)
    {
        path_ = create_unique_test_directory("ncnn_moe_attention_test_");
        const uint32_t hidden_size = grouped_query ? 4 : 2;
        const uint32_t attention_head_count = grouped_query ? 2 : 1;

        std::ofstream manifest(path_ / "config.json", std::ios::binary);
        manifest << R"({
  "model_type": "test_moe",
  "vocabulary_size": 2,
  "hidden_size": )"
                 << hidden_size << R"(,
  "intermediate_size": 2,
  "layer_count": 1,
  "expert_count": 1,
  "experts_per_token": 1,
  "expert_activation": "relu",
  "expert_layout": "up_down",
  "normalize_topk_weights": true,
  "use_attention": true,
  "attention_head_count": )"
                 << attention_head_count << R"(,
  "kv_head_count": 1,
  "head_dimension": 2,
  "sliding_window": )"
                 << sliding_window << R"(,
  "initial_context_length": 16,
  "max_context_length": )"
                 << (context_length != 0 ? context_length : grouped_query ? 64
                                                                          : 32)
                 << R"(,
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
        if (!grouped_query)
        {
            append({1.0f, 0.0f, 0.0f, 1.0f});
            append({1.0f, 1.0f});
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
            append({1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f});
            append({1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f});
        }
        else
        {
            append({1.0f, 0.25f, -0.5f, 0.125f, -0.5f, 1.0f, 0.25f, 0.75f});
            append({1.0f, 1.0f, 1.0f, 1.0f});
            append({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
            if (attention_bias)
                append({0.0f, 0.0f, 0.0f, 0.0f});
            append({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
            if (attention_bias)
                append({0.0f, 0.0f});
            append({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
            if (attention_bias)
                append({0.0f, 0.0f});
            append({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
            if (attention_bias)
                append({0.0f, 0.0f, 0.0f, 0.0f});
            if (attention_sinks)
                append({0.375f, -0.125f});
            append({1.0f, 1.0f, 1.0f, 1.0f});
            append({1.0f, 0.5f, -0.25f, 0.125f});
            append({0.5f, 0.25f, 0.125f, -0.25f, -0.125f, 0.5f, 0.25f, 0.125f});
            append({0.5f, 0.125f, 0.25f, -0.5f, -0.125f, 0.25f, 0.5f, 0.25f});
            append({1.0f, 1.0f, 1.0f, 1.0f});
            append({1.0f, 0.5f, -0.5f, 0.25f, -0.5f, 0.25f, 0.5f, 1.0f});
        }
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
// Inject a failure after Attention has updated KV, without a production fault hook.
// The model is restored before any recovery checks or further scheduler submissions.
class ScopedLmHeadFailure
{
public:
    ScopedLmHeadFailure(const Model& model, bool exception)
        : compiled(const_cast<CompiledModel&>(model_compiled(model))),
          throw_exception(exception)
    {
        for (Layer& candidate : compiled.net.layers)
        {
            if (candidate.kind == LayerKind::LmHead)
            {
                layer = &candidate;
                break;
            }
        }
        check(layer != nullptr);
        if (throw_exception)
        {
            TensorData& weight = compiled.weights.at_mutable(layer->weights.front());
            saved_weight = weight;
            check(weight.dtype == DType::Float32);
            weight.float32_data.clear();
            weight.mapped_data.reset();
            weight.mapped_size = 0;
        }
        else
        {
            saved_bindings.swap(layer->weights);
        }
    }

    ~ScopedLmHeadFailure()
    {
        if (throw_exception)
            compiled.weights.at_mutable(layer->weights.front()) = std::move(saved_weight);
        else
            saved_bindings.swap(layer->weights);
    }

private:
    CompiledModel& compiled;
    Layer* layer = nullptr;
    bool throw_exception;
    TensorData saved_weight;
    std::vector<TensorHandle> saved_bindings;
};

static void check_session_requires_reset(const SessionPtr& session, uint64_t prefill_tokens = 0)
{
    check(session->sequence_length() == prefill_tokens);
    check(session->statistics().prefill_tokens == prefill_tokens);
    check(session->statistics().decode_tokens == 0);
    auto decoded = session->decode(1);
    check(!decoded);
    check(decoded.error().code == ErrorCode::InternalError);
    check(decoded.error().message.find("reset") != std::string::npos);
    const std::array<int32_t, 1> input = {0};
    auto prefilled = session->prefill(input);
    check(!prefilled);
    check(prefilled.error().code == ErrorCode::InternalError);
    GenerationOptions options;
    options.max_new_tokens = 1;
    options.use_speculative = false;
    auto generated = session->generate(input, options);
    check(!generated);
    check(generated.error().code == ErrorCode::InternalError);
}

static TensorHandle add_float_tensor(WeightStore& weights,
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

static TensorHandle add_bfloat16_tensor(WeightStore& weights,
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

static TensorHandle add_int64_tensor(WeightStore& weights,
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

ModelPackage qwen4_exp_package();
WeightMapping qwen4_exp_test_mapping(const MoeModelDescriptor& descriptor);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_TEST_RUNTIME_SUPPORT_H
