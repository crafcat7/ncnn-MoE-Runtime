#include "models/modeladapter_qwen3.h"
#include "backends/ncnn/gpusegment_vulkan.h"
#include "backends/ncnn/vulkan.h"
#include "graph/compiler.h"
#include "engine/executor.h"
#include "engine/sessionstate.h"
#include "kernels/ops.h"
#include "ncnn/moe/runtime.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ncnn {
namespace moe {

static void check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static std::string config_json(bool tied)
{
    // Qwen3-0.6B also has attention_head_count * head_dimension != hidden_size.
    return std::string(R"({"model_type":"qwen3","vocab_size":8,"hidden_size":4,
        "intermediate_size":6,"num_hidden_layers":2,"num_attention_heads":2,
        "num_key_value_heads":1,"head_dim":4,"max_position_embeddings":32,
        "rms_norm_eps":0.000001,"rope_theta":1000000.0,"rope_scaling":null,
        "hidden_act":"silu","torch_dtype":"float32","attention_bias":false,
        "use_sliding_window":false,"sliding_window":null,"tie_word_embeddings":)")
           + (tied ? "true" : "false") + "}";
}

struct FixtureTensor
{
    std::string name;
    std::vector<uint32_t> shape;
    std::vector<float> values;
};

static std::vector<FixtureTensor> fixture_tensors(bool tied)
{
    std::vector<FixtureTensor> tensors = {
        {"model.embed_tokens.weight", {8, 4}, {}},
        {"model.norm.weight", {4}, {}},
    };
    if (!tied)
        tensors.push_back({"lm_head.weight", {8, 4}, {}});
    for (uint32_t layer = 0; layer < 2; ++layer)
    {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        for (const FixtureTensor& tensor : std::vector<FixtureTensor>{
                 {"input_layernorm.weight", {4}, {}},
                 {"post_attention_layernorm.weight", {4}, {}},
                 {"self_attn.q_proj.weight", {8, 4}, {}},
                 {"self_attn.k_proj.weight", {4, 4}, {}},
                 {"self_attn.v_proj.weight", {4, 4}, {}},
                 {"self_attn.o_proj.weight", {4, 8}, {}},
                 {"self_attn.q_norm.weight", {4}, {}},
                 {"self_attn.k_norm.weight", {4}, {}},
                 {"mlp.gate_proj.weight", {6, 4}, {}},
                 {"mlp.up_proj.weight", {6, 4}, {}},
                 {"mlp.down_proj.weight", {4, 6}, {}},
             })
            tensors.push_back({prefix + tensor.name, tensor.shape, {}});
    }
    for (size_t index = 0; index < tensors.size(); ++index)
    {
        FixtureTensor& tensor = tensors[index];
        size_t count = 1;
        for (uint32_t size : tensor.shape)
            count *= size;
        tensor.values.resize(count);
        const bool norm = tensor.shape.size() == 1;
        for (size_t element = 0; element < count; ++element)
        {
            const int pattern = static_cast<int>((element * 7 + index * 3 + element / 4) % 19) - 9;
            tensor.values[element] = (norm ? 1.0f : 0.0f) + static_cast<float>(pattern) * (norm ? 0.03125f : 0.0625f);
        }
    }
    return tensors;
}

static void write_weights(const std::filesystem::path& root, const std::vector<FixtureTensor>& tensors)
{
    std::ostringstream header;
    std::vector<float> values;
    header << '{';
    for (size_t index = 0; index < tensors.size(); ++index)
    {
        const FixtureTensor& tensor = tensors[index];
        if (index != 0)
            header << ',';
        header << '"' << tensor.name << R"(":{"dtype":"F32","shape":[)";
        for (size_t axis = 0; axis < tensor.shape.size(); ++axis)
        {
            if (axis != 0)
                header << ',';
            header << tensor.shape[axis];
        }
        const size_t begin = values.size() * sizeof(float);
        values.insert(values.end(), tensor.values.begin(), tensor.values.end());
        header << "],\"data_offsets\":[" << begin << ',' << values.size() * sizeof(float) << "]}";
    }
    header << '}';
    std::string json = header.str();
    json.append((8 - json.size() % 8) % 8, ' ');
    const uint64_t size = json.size();
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    output.write(reinterpret_cast<const char*>(&size), sizeof(size));
    output.write(json.data(), static_cast<std::streamsize>(json.size()));
    output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    check(static_cast<bool>(output), "cannot write Qwen3 weights");
}

static const FixtureTensor& find_tensor(const std::vector<FixtureTensor>& tensors, const std::string& name)
{
    const auto found = std::find_if(tensors.begin(), tensors.end(), [&](const FixtureTensor& tensor) { return tensor.name == name; });
    check(found != tensors.end(), "missing reference tensor: " + name);
    return *found;
}

static std::vector<float> project(const FixtureTensor& weight, const std::vector<float>& input)
{
    check(weight.shape.size() == 2 && weight.shape[1] == input.size(), "reference projection shape mismatch");
    std::vector<float> output(weight.shape[0], 0.0f);
    for (size_t row = 0; row < output.size(); ++row)
        for (size_t column = 0; column < input.size(); ++column)
            output[row] += weight.values[row * input.size() + column] * input[column];
    return output;
}

static void normalize(std::vector<float>& values, const FixtureTensor& weight)
{
    const size_t dimension = weight.values.size();
    for (size_t begin = 0; begin < values.size(); begin += dimension)
    {
        float square_sum = 0.0f;
        for (size_t column = 0; column < dimension; ++column)
            square_sum += values[begin + column] * values[begin + column];
        const float scale = 1.0f / std::sqrt(square_sum / static_cast<float>(dimension) + 1e-6f);
        for (size_t column = 0; column < dimension; ++column)
            values[begin + column] *= scale * weight.values[column];
    }
}

static void rotate(std::vector<float>& values, size_t position)
{
    for (size_t begin = 0; begin < values.size(); begin += 4)
        for (size_t column = 0; column < 2; ++column)
        {
            const float angle = static_cast<float>(position) / std::pow(1000000.0f, static_cast<float>(column) / 2.0f);
            const float first = values[begin + column];
            const float second = values[begin + column + 2];
            values[begin + column] = first * std::cos(angle) - second * std::sin(angle);
            values[begin + column + 2] = second * std::cos(angle) + first * std::sin(angle);
        }
}

static std::vector<std::vector<float>> reference_logits(const std::vector<FixtureTensor>& tensors, const std::vector<int32_t>& ids, bool tied)
{
    const auto tensor = [&](const std::string& name) -> const FixtureTensor& { return find_tensor(tensors, name); };
    std::vector<std::vector<float>> hidden;
    for (int32_t id : ids)
    {
        const auto& embedding = tensor("model.embed_tokens.weight").values;
        hidden.emplace_back(embedding.begin() + id * 4, embedding.begin() + id * 4 + 4);
    }
    for (uint32_t layer = 0; layer < 2; ++layer)
    {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        std::vector<std::vector<float>> queries, keys, values;
        for (size_t position = 0; position < ids.size(); ++position)
        {
            auto normalized = hidden[position];
            normalize(normalized, tensor(prefix + "input_layernorm.weight"));
            auto query = project(tensor(prefix + "self_attn.q_proj.weight"), normalized);
            auto key = project(tensor(prefix + "self_attn.k_proj.weight"), normalized);
            normalize(query, tensor(prefix + "self_attn.q_norm.weight"));
            normalize(key, tensor(prefix + "self_attn.k_norm.weight"));
            rotate(query, position);
            rotate(key, position);
            queries.push_back(std::move(query));
            keys.push_back(std::move(key));
            values.push_back(project(tensor(prefix + "self_attn.v_proj.weight"), normalized));
        }
        for (size_t position = 0; position < ids.size(); ++position)
        {
            std::vector<float> attention(8, 0.0f);
            for (size_t head = 0; head < 2; ++head)
            {
                std::vector<float> scores(position + 1, 0.0f);
                for (size_t previous = 0; previous <= position; ++previous)
                    for (size_t column = 0; column < 4; ++column)
                        scores[previous] += queries[position][head * 4 + column] * keys[previous][column] * 0.5f;
                const float maximum = *std::max_element(scores.begin(), scores.end());
                float sum = 0.0f;
                for (float& score : scores)
                {
                    score = std::exp(score - maximum);
                    sum += score;
                }
                for (size_t previous = 0; previous <= position; ++previous)
                    for (size_t column = 0; column < 4; ++column)
                        attention[head * 4 + column] += scores[previous] / sum * values[previous][column];
            }
            const auto output = project(tensor(prefix + "self_attn.o_proj.weight"), attention);
            for (size_t column = 0; column < 4; ++column)
                hidden[position][column] += output[column];
            auto normalized = hidden[position];
            normalize(normalized, tensor(prefix + "post_attention_layernorm.weight"));
            auto gate = project(tensor(prefix + "mlp.gate_proj.weight"), normalized);
            const auto up = project(tensor(prefix + "mlp.up_proj.weight"), normalized);
            for (size_t column = 0; column < gate.size(); ++column)
                gate[column] = gate[column] / (1.0f + std::exp(-gate[column])) * up[column];
            const auto down = project(tensor(prefix + "mlp.down_proj.weight"), gate);
            for (size_t column = 0; column < 4; ++column)
                hidden[position][column] += down[column];
        }
    }
    std::vector<std::vector<float>> logits;
    for (auto& row : hidden)
    {
        normalize(row, tensor("model.norm.weight"));
        logits.push_back(project(tensor(tied ? "model.embed_tokens.weight" : "lm_head.weight"), row));
    }
    return logits;
}

static void check_logits(std::span<const float> actual, const std::vector<float>& expected, const std::string& context)
{
    check(actual.size() == expected.size(), context + " logit count mismatch");
    for (size_t index = 0; index < actual.size(); ++index)
    {
        if (std::abs(actual[index] - expected[index]) <= 2e-5f + 2e-5f * std::abs(expected[index]))
            continue;
        std::ostringstream message;
        message << context << " index=" << index << " actual=" << actual[index] << " expected=" << expected[index];
        throw std::runtime_error(message.str());
    }
}

static void check_execution(const CompiledModel& model, const std::vector<FixtureTensor>& tensors, bool tied)
{
    const std::vector<int32_t> ids = {1, 2, 4, 3};
    const auto expected = reference_logits(tensors, ids, tied);
    SessionState full;
    SessionStatistics statistics;
    auto status = forward_model(model, ids, statistics, full, 0, LogitsOutput::All);
    check(static_cast<bool>(status), status ? "" : status.error().message);
    check(full.logits.rows() == ids.size(), "prefill did not return every token's logits");
    for (size_t row = 0; row < ids.size(); ++row)
        check_logits({full.logits.row(row), full.logits.columns()}, expected[row], "scalar prefill reference");
    check(statistics.expert_assignments == 0 && statistics.expert_batches == 0, "dense prefill executed MoE routes");

    // One-token steps cover the decode shape; multi-row chunks are exercised
    // below by the public Session and the resident GPU chain.
    for (size_t chunk_size : {size_t(1)})
    {
        SessionState chunked;
        SessionStatistics chunked_statistics;
        for (size_t begin = 0; begin < ids.size(); begin += chunk_size)
        {
            const std::span<const int32_t> chunk(ids.data() + begin, std::min(chunk_size, ids.size() - begin));
            status = forward_model(model, chunk, chunked_statistics, chunked, begin, LogitsOutput::All);
            check(static_cast<bool>(status), status ? "" : status.error().message);
            for (size_t row = 0; row < chunk.size(); ++row)
                check_logits({chunked.logits.row(row), chunked.logits.columns()}, expected[begin + row], "scalar decode/chunked reference");
        }
    }

    std::array<SessionState, 2> states;
    std::array<SessionStatistics, 2> batch_statistics;
    std::array<std::vector<float>, 2> outputs;
    std::array<DecodeBatchEntry, 2> entries;
    const std::array<std::vector<int32_t>, 2> sequences = {{{1, 2, 3}, {4, 1, 2, 5}}};
    for (size_t index = 0; index < entries.size(); ++index)
    {
        const auto& sequence = sequences[index];
        const std::span<const int32_t> prefix(sequence.data(), sequence.size() - 1);
        status = forward_model(model, prefix, batch_statistics[index], states[index], 0, LogitsOutput::Last);
        check(static_cast<bool>(status), status ? "" : status.error().message);
        entries[index] = {sequence.back(), &batch_statistics[index], &states[index], prefix.size(), &outputs[index]};
    }
    BatchWorkspace workspace;
    status = forward_decode_batch(model, entries, workspace);
    check(static_cast<bool>(status), status ? "" : status.error().message);
    for (size_t index = 0; index < entries.size(); ++index)
    {
        check_logits(outputs[index], reference_logits(tensors, sequences[index], tied).back(), "scalar batch decode reference");
        check(batch_statistics[index].expert_assignments == 0 && batch_statistics[index].expert_batches == 0,
              "dense batch executed MoE routes");
    }
}

static void test_fixture(const std::filesystem::path& root, bool tied)
{
    std::filesystem::create_directories(root);
    ModelPackage package;
    package.root = root;
    package.manifest.model_type = "qwen3";
    package.manifest.raw_json = config_json(tied);
    std::ofstream(root / "config.json") << package.manifest.raw_json;
    const auto tensors = fixture_tensors(tied);
    write_weights(root, tensors);
    Qwen3Adapter adapter;
    check(adapter.can_load(package.manifest), "Qwen3 adapter did not admit qwen3");
    auto descriptor = adapter.parse_model(package);
    check(static_cast<bool>(descriptor), descriptor ? "" : descriptor.error().message);
    const auto& desc = descriptor.value();
    check(desc.model_type == "qwen3" && desc.layers.size() == 2 && desc.hidden_size == 4, "incorrect Qwen3 model dimensions");
    check(desc.attention_head_count == 2 && desc.kv_head_count == 1 && desc.head_dimension == 4,
          "Qwen3 explicit head dimension or GQA was lost");
    check(desc.expert_count == 0 && desc.experts_per_token == 0 && desc.norm_weight_offset == 0.0f,
          "Qwen3 inherited MoE counts or Qwen3.5 norm offset");
    check(desc.kv_cache_dtype == DType::Float32, "F32 Qwen3 did not retain F32 KV");
    for (const auto& layer : desc.layers)
    {
        check(layer.dense.enabled() && layer.dense.intermediate_size == 6 && layer.moe.expert_count == 0,
              "Qwen3 layer is not a native dense FFN");
        check(layer.attention.kind == AttentionKind::Standard && layer.attention.flags == AttentionDescriptorQueryKeyNorm,
              "Qwen3 attention did not use plain GQA with QK norm");
        check(layer.attention.max_context_length == 32 && layer.attention.rope_theta == 1000000.0f,
              "Qwen3 context or RoPE theta was lost");
    }
    auto mapping = adapter.map_weights(package, desc);
    check(static_cast<bool>(mapping), mapping ? "" : mapping.error().message);
    check(mapping.value().at("lm_head.weight").float32_values().size() == 32, "missing Qwen3 LM head");
    if (tied)
        check(std::equal(mapping.value().at("lm_head.weight").float32_values().begin(),
                         mapping.value().at("lm_head.weight").float32_values().end(),
                         mapping.value().at("token_embedding.weight").float32_values().begin()),
              "tied Qwen3 head did not use token embeddings");
    CompilerOption opt;
    opt.flags = BackendCpuExecution;
    opt.optimization_flags = 0;
    CompiledModel compiled;
    auto status = compile_model(desc, mapping.value(), compiled, HybridMode::CpuOnly, opt);
    check(static_cast<bool>(status), status ? "" : status.error().message);
    size_t dense_count = 0;
    for (const auto& node : compiled.graph.nodes)
    {
        dense_count += node.type == ExecutionNodeType::DenseFfn;
        check(node.type != ExecutionNodeType::Router && node.type != ExecutionNodeType::ExpertDispatch
                  && node.type != ExecutionNodeType::ExpertGroup && node.type != ExecutionNodeType::Combine,
              "Qwen3 compiled a MoE graph node");
    }
    check(dense_count == 2, "Qwen3 did not compile one dense FFN per layer");
    check_execution(compiled, tensors, tied);

    Runtime runtime;
    Option runtime_opt;
    runtime_opt.hybrid_mode = HybridMode::CpuOnly;
    auto loaded = runtime.load_model(root, runtime_opt);
    check(static_cast<bool>(loaded), loaded ? "" : loaded.error().message);
    auto session = runtime.create_session(loaded.value());
    check(static_cast<bool>(session), session ? "" : session.error().message);
    const std::vector<int32_t> ids = {1, 2, 4};
    auto prefill = session.value()->prefill(ids);
    check(static_cast<bool>(prefill), prefill ? "" : prefill.error().message);
    check_logits(prefill.value().logits, reference_logits(tensors, ids, tied).back(), "registered runtime adapter");

    ModelPackage missing_package = package;
    missing_package.root = root / "missing_norm";
    std::filesystem::create_directories(missing_package.root);
    auto missing = tensors;
    missing.erase(std::remove_if(missing.begin(), missing.end(), [](const FixtureTensor& tensor) {
                      return tensor.name == "model.layers.0.self_attn.q_norm.weight";
                  }),
                  missing.end());
    write_weights(missing_package.root, missing);
    check(!adapter.map_weights(missing_package, desc), "Qwen3 silently accepted missing QK norm");

    missing_package.root = root / "wrong_query_shape";
    std::filesystem::create_directories(missing_package.root);
    auto wrong_shape = tensors;
    for (auto& tensor : wrong_shape)
        if (tensor.name == "model.layers.0.self_attn.q_proj.weight")
            tensor.shape = {4, 8};
    write_weights(missing_package.root, wrong_shape);
    auto invalid_mapping = adapter.map_weights(missing_package, desc);
    if (invalid_mapping)
    {
        CompiledModel invalid_model;
        check(!compile_model(desc, invalid_mapping.value(), invalid_model, HybridMode::CpuOnly, opt),
              "Qwen3 silently accepted transposed query projection shape");
    }
}

static void test_invalid_config()
{
    Qwen3Adapter adapter;
    const std::array<std::pair<std::string, std::string>, 10> changes = {{
        {"\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\""},
        {"\"attention_bias\":false", "\"attention_bias\":true"},
        {"\"use_sliding_window\":false", "\"use_sliding_window\":true"},
        {"\"rope_scaling\":null", "\"rope_scaling\":{\"rope_type\":\"linear\",\"factor\":2}"},
        {"\"num_key_value_heads\":1", "\"num_key_value_heads\":3"},
        {"\"head_dim\":4", "\"head_dim\":3"},
        {"\"intermediate_size\":6", "\"intermediate_size\":0"},
        {"\"num_hidden_layers\":2", "\"num_hidden_layers\":2,\"layer_types\":[\"full_attention\",\"sliding_attention\"]"},
        {"\"num_hidden_layers\":2", "\"num_hidden_layers\":2,\"layer_types\":[\"full_attention\"]"},
        {"\"num_hidden_layers\":2", "\"num_hidden_layers\":2,\"layer_types\":null"},
    }};
    for (const auto& change : changes)
    {
        ModelPackage package;
        package.manifest.model_type = "qwen3";
        package.manifest.raw_json = config_json(true);
        const size_t offset = package.manifest.raw_json.find(change.first);
        check(offset != std::string::npos, "invalid-config test replacement failed");
        package.manifest.raw_json.replace(offset, change.first.size(), change.second);
        check(!adapter.parse_model(package), "Qwen3 accepted unsupported config: " + change.second);
    }
    ModelPackage explicit_layers;
    explicit_layers.manifest.model_type = "qwen3";
    explicit_layers.manifest.raw_json = config_json(true);
    explicit_layers.manifest.raw_json.insert(1, "\"layer_types\":[\"full_attention\",\"full_attention\"],");
    auto descriptor = adapter.parse_model(explicit_layers);
    check(static_cast<bool>(descriptor), descriptor ? "" : descriptor.error().message);
    check(descriptor.value().layers.size() == 2
              && std::all_of(descriptor.value().layers.begin(), descriptor.value().layers.end(), [](const LayerDescriptor& layer) {
                     return layer.attention.kind == AttentionKind::Standard && layer.attention.sliding_window == 0;
                 }),
          "Qwen3 explicit full-attention layer types were not preserved");
    ModelManifest manifest;
    manifest.model_type = "qwen3_moe";
    check(!adapter.can_load(manifest), "dense Qwen3 adapter claimed qwen3_moe");
}

static void test_gpu_fixture(const std::filesystem::path& root, uint32_t device_index)
{
    std::filesystem::create_directories(root);
    const auto tensors = fixture_tensors(true);
    std::string config = config_json(true);
    const auto context_offset = config.find("\"max_position_embeddings\":32");
    config.replace(context_offset, std::string("\"max_position_embeddings\":32").size(), "\"max_position_embeddings\":1024");
    std::ofstream(root / "config.json") << config;
    write_weights(root, tensors);
    Runtime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::HybridExperts;
    opt.vulkan_device_index = device_index;
    opt.optimization_flags &= ~OptimizationCpuFastSilu;
    auto loaded = runtime.load_model(root, opt);
    check(static_cast<bool>(loaded), loaded ? "" : loaded.error().message);
    const auto& compiled = model_compiled(*loaded.value());
    check(compiled.resident_submission_rows > 0, "supported dense model did not select continuous GPU execution");
    check(!compiled.expert_backend, "resident dense model allocated a routed Expert cache");
    for (const auto& layer : compiled.graph.layer_plans)
        check(!layer.dense_ffn.use_gpu_cache, "dense FFN remained in the evictable Expert cache");
    const auto before = get_vulkan_statistics(compiled.vulkan_runtime);
    check_execution(compiled, tensors, true);
    const auto after = get_vulkan_statistics(compiled.vulkan_runtime);
    check(after.attention_blocks > before.attention_blocks && after.resident_graph_submissions > before.resident_graph_submissions,
          "dense GPU test did not execute resident Attention/FFN work");
    check(after.attention_cpu_fallbacks == before.attention_cpu_fallbacks,
          "resident dense Attention fell back to the CPU");
    check(after.weight_upload_bytes == before.weight_upload_bytes
              && after.weight_transfer_submissions == before.weight_transfer_submissions,
          "resident dense inference reuploaded model weights");
    // Eight forwards above cover all-logits prefill, single-token decode and a
    // mixed-length batch. Each uploads hidden once and downloads only logits.
    check(after.batch_uploads - before.batch_uploads == 8
              && after.batch_downloads - before.batch_downloads == 8,
          "dense GPU execution crossed a host boundary between layers");

    // More than half the command row budget forces two recording chunks;
    // hidden must still remain on the GPU across their fence.
    SessionOptions session_opt;
    session_opt.prefill_chunk_size = 512;
    auto session = runtime.create_session(loaded.value(), session_opt);
    check(static_cast<bool>(session), session ? "" : session.error().message);
    std::vector<int32_t> prompt(257);
    for (size_t i = 0; i < prompt.size(); ++i)
        prompt[i] = static_cast<int32_t>(i % 7 + 1);
    const auto before_long = get_vulkan_statistics(compiled.vulkan_runtime);
    auto prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill), prefill ? "" : prefill.error().message);
    check_logits(prefill.value().logits, reference_logits(tensors, prompt, true).back(), "multi-submit dense prefill");
    const auto after_long = get_vulkan_statistics(compiled.vulkan_runtime);
    check(after_long.resident_graph_submissions - before_long.resident_graph_submissions >= 2
              && after_long.batch_uploads - before_long.batch_uploads == 1
              && after_long.batch_downloads - before_long.batch_downloads == 1,
          "dense command chunks did not preserve device hidden across submissions: submits="
              + std::to_string(after_long.resident_graph_submissions - before_long.resident_graph_submissions)
              + " uploads=" + std::to_string(after_long.batch_uploads - before_long.batch_uploads)
              + " downloads=" + std::to_string(after_long.batch_downloads - before_long.batch_downloads));

    // Public Session chunking suppresses logits for intermediate chunks.
    session_opt.prefill_chunk_size = 2;
    session = runtime.create_session(loaded.value(), session_opt);
    check(static_cast<bool>(session), session ? "" : session.error().message);
    prompt = {1, 2, 4, 3, 5};
    prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill), prefill ? "" : prefill.error().message);
    check_logits(prefill.value().logits, reference_logits(tensors, prompt, true).back(), "public chunked dense prefill");
    auto decoded = session.value()->decode(6);
    check(static_cast<bool>(decoded), decoded ? "" : decoded.error().message);
    prompt.push_back(6);
    check_logits(decoded.value().logits, reference_logits(tensors, prompt, true).back(), "public dense decode");
    auto reset = session.value()->reset();
    check(static_cast<bool>(reset), reset ? "" : reset.error().message);
    prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill), prefill ? "" : prefill.error().message);
    check_logits(prefill.value().logits, reference_logits(tensors, prompt, true).back(), "reset dense session");

    // A later chunk failure must not publish partial logits or leave earlier
    // submitted in-place KV state available for an implicit retry.
    CompiledModel rejected = compiled;
    rejected.resident_submission_rows = 1;
    const std::array<int32_t, 1> ids = {1};
    ActivationBuffer input;
    forward_embedding(rejected.weights.at(rejected.token_embedding), ids, input);
    std::array<LayerCache, 2> caches;
    std::array<LayerCache*, 2> pointers = {&caches[0], &caches[1]};
    caches[1].token_count = 1;
    caches[1].capacity_tokens = 0;
    const std::array<uint64_t, 1> positions = {0};
    ActivationBuffer logits(1, 1);
    logits.row(0)[0] = -73.0f;
    GpuSegmentWorkspace_vulkan workspace;
    std::string error;
    check(!GpuSegment_vulkan::forward_resident(rejected, input, positions, pointers, {}, true, true, logits, error, &workspace),
          "late dense recording failure was accepted");
    check(logits.rows() == 1 && logits.columns() == 1 && logits.row(0)[0] == -73.0f,
          "failed dense chain overwrote published logits");
    for (const auto& cache : caches)
        check(cache.vulkan_attention_state_unknown && cache.token_count == 1,
              "failed submitted dense chain did not invalidate partially advanced KV state");
    const auto failed = get_vulkan_statistics(compiled.vulkan_runtime);
    check(!GpuSegment_vulkan::forward_resident(rejected, input, positions, pointers, {}, true, true, logits, error, &workspace)
              && get_vulkan_statistics(compiled.vulkan_runtime).compute_submissions == failed.compute_submissions,
          "dense chain retried invalidated cache state without reset");
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    const bool vulkan = argc == 2 && std::string_view(argv[1]) == "--vulkan";
    const auto gpus = vulkan ? ncnn::moe::get_gpu_infos() : std::vector<ncnn::moe::GpuInfo>{};
    if (vulkan && gpus.empty())
    {
        std::cout << "SKIP Qwen3 Vulkan: no Vulkan device is available\n";
        return 77;
    }
    const auto root = std::filesystem::temp_directory_path()
                      / ("ncnn_moe_qwen3_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try
    {
        if (vulkan)
        {
            for (const auto& gpu : gpus)
                ncnn::moe::test_gpu_fixture(root / ("gpu-" + std::to_string(gpu.device_index)), gpu.device_index);
        }
        else
        {
            ncnn::moe::test_invalid_config();
            ncnn::moe::test_fixture(root / "tied", true);
            ncnn::moe::test_fixture(root / "untied", false);
        }
        std::filesystem::remove_all(root);
        std::cout << "Qwen3 adapter, scalar logits, prefill, decode, chunking and batch tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}
