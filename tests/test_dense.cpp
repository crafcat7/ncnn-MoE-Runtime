#include "engine/executor.h"
#include "engine/expert.h"
#include "engine/sessionstate.h"
#include "graph/compiler.h"
#include "models/tensornames.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void dense_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(std::string("dense FFN test failed: ") + message);
}

static void dense_near(float actual, float expected)
{
    dense_check(std::abs(actual - expected) < 1e-5f, "scalar reference mismatch");
}

static TensorData dense_tensor(std::vector<uint32_t> shape, uint32_t seed, DType dtype = DType::Float32)
{
    TensorData tensor;
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    for (uint64_t index = 0; index < tensor.element_count(); ++index)
    {
        const float value = static_cast<float>(static_cast<int>((index * 5 + seed) % 19) - 9) / 32.0f;
        if (dtype == DType::BFloat16)
            tensor.bfloat16_data.push_back(static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16));
        else
            tensor.float32_data.push_back(value);
    }
    return tensor;
}

static MoeModelDescriptor dense_descriptor(DType dtype)
{
    MoeModelDescriptor descriptor;
    descriptor.model_type = "dense-test";
    descriptor.vocabulary_size = 7;
    descriptor.hidden_size = 4;
    descriptor.intermediate_size = 5;
    descriptor.layers.resize(2);
    descriptor.layers[0].dense.intermediate_size = 5;
    descriptor.layers[1].dense.intermediate_size = 3;
    for (LayerDescriptor& layer : descriptor.layers)
    {
        layer.dense.gate_weight_dtype = dtype;
        layer.dense.up_weight_dtype = dtype;
        layer.dense.down_weight_dtype = dtype;
    }
    return descriptor;
}

static WeightMapping dense_mapping(const MoeModelDescriptor& descriptor)
{
    WeightMapping mapping;
    mapping.emplace("token_embedding.weight", dense_tensor({7, 4}, 3));
    mapping.emplace("lm_head.weight", dense_tensor({7, 4}, 11));
    TensorData norm;
    norm.shape = {4};
    norm.float32_data = {0.75f, 1.0f, 1.25f, 1.5f};
    mapping.emplace("final_norm.weight", norm);
    for (uint32_t index = 0; index < descriptor.layers.size(); ++index)
    {
        const DenseDescriptor& dense = descriptor.layers[index].dense;
        const std::string prefix = layer_prefix(index);
        mapping.emplace(prefix + "pre_ffn_norm.weight", norm);
        mapping.emplace(prefix + "ffn.gate.weight", dense_tensor({dense.intermediate_size, 4}, index + 1, dense.gate_weight_dtype));
        mapping.emplace(prefix + "ffn.up.weight", dense_tensor({dense.intermediate_size, 4}, index + 4, dense.up_weight_dtype));
        mapping.emplace(prefix + "ffn.down.weight", dense_tensor({4, dense.intermediate_size}, index + 9, dense.down_weight_dtype));
    }
    return mapping;
}

static float dense_value(const TensorData& tensor, size_t index)
{
    return tensor.dtype == DType::BFloat16
               ? std::bit_cast<float>(static_cast<uint32_t>(tensor.bfloat16_values()[index]) << 16)
               : tensor.float32_values()[index];
}

static std::vector<float> dense_normalize(const std::vector<float>& input, const TensorData& norm, float epsilon)
{
    float sum = 0.0f;
    for (float value : input)
        sum += value * value;
    const float scale = 1.0f / std::sqrt(sum / static_cast<float>(input.size()) + epsilon);
    std::vector<float> result(input.size());
    for (size_t index = 0; index < input.size(); ++index)
        result[index] = input[index] * scale * dense_value(norm, index);
    return result;
}

static std::vector<float> dense_project(const TensorData& matrix, const std::vector<float>& input)
{
    std::vector<float> result(matrix.shape[0], 0.0f);
    for (uint32_t row = 0; row < matrix.shape[0]; ++row)
        for (uint32_t column = 0; column < matrix.shape[1]; ++column)
            result[row] += dense_value(matrix, static_cast<size_t>(row) * matrix.shape[1] + column) * input[column];
    return result;
}

static std::vector<float> dense_reference(const CompiledModel& model, int32_t token)
{
    const TensorData& embedding = model.weights.at(model.token_embedding);
    std::vector<float> hidden(4);
    for (uint32_t column = 0; column < hidden.size(); ++column)
        hidden[column] = dense_value(embedding, static_cast<size_t>(token) * 4 + column);
    for (const CompiledLayerPlan& layer : model.graph.layer_plans)
    {
        const std::vector<float> normalized = dense_normalize(hidden, model.weights.at(layer.dense_pre_ffn_norm_weight), model.descriptor.norm_epsilon);
        std::vector<float> gate = dense_project(model.weights.at(layer.dense_ffn.gate_weight), normalized);
        const std::vector<float> up = dense_project(model.weights.at(layer.dense_ffn.up_weight), normalized);
        for (size_t index = 0; index < gate.size(); ++index)
            gate[index] = gate[index] / (1.0f + std::exp(-gate[index])) * up[index];
        const std::vector<float> output = dense_project(model.weights.at(layer.dense_ffn.down_weight), gate);
        for (size_t column = 0; column < hidden.size(); ++column)
            hidden[column] += output[column];
    }
    return dense_project(model.weights.at(model.lm_head_weight), dense_normalize(hidden, model.weights.at(model.final_norm_weight), model.descriptor.norm_epsilon));
}

static void dense_no_routes(const SessionStatistics& statistics)
{
    dense_check(statistics.expert_assignments == 0 && statistics.expert_batches == 0, "dense FFN recorded routes");
    dense_check(statistics.expert_cache_hits == 0 && statistics.expert_cache_misses == 0, "dense FFN accessed expert cache");
    dense_check(statistics.expert_token_counts.empty(), "dense FFN allocated expert counts");
}

static TensorData native_matrix(uint32_t rows, uint32_t columns, bool packed, uint32_t seed)
{
    TensorData tensor;
    tensor.shape = {rows, columns};
    tensor.dtype = packed ? DType::Int4 : DType::Float32;
    if (packed)
    {
        tensor.quantization_group_size = columns;
        tensor.quantization_scales.assign(rows, 0.25f);
        tensor.quantization_zero_points.assign(rows, 8);
        tensor.quantized_data.assign(static_cast<size_t>(rows) * ((columns + 1) / 2), 0);
    }
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            const uint8_t code = static_cast<uint8_t>((row * 3 + column * 5 + seed) % 16);
            if (packed)
            {
                const size_t offset = static_cast<size_t>(row) * ((columns + 1) / 2) + column / 2;
                tensor.quantized_data[offset] |= code << ((column % 2) * 4);
            }
            else
                tensor.float32_data.push_back((static_cast<int>(code) - 8) * 0.25f);
        }
    }
    return tensor;
}

static TensorHandle native_add(CompiledModel& model, const std::string& name, TensorData tensor)
{
    auto handle = model.weights.add(name, std::move(tensor));
    dense_check(static_cast<bool>(handle), "native expert weight insertion");
    return handle.value();
}

static void test_native_expert_layouts()
{
    for (bool packed : {false, true})
    {
        for (ExpertActivation activation : {ExpertActivation::Silu, ExpertActivation::DeepSeekSwiGlu,
                                            ExpertActivation::GptOssSwiGlu})
        {
            for (ExpertLayout layout : {ExpertLayout::GateUpDown, ExpertLayout::PackedGateUpDown,
                                        ExpertLayout::InterleavedGateUpDown})
            {
                CompiledModel model;
                model.opt.optimization_flags = 0;
                ExpertPlan expert;
                expert.layout = layout;
                expert.activation = activation;
                expert.activation_limit = 0.75f;
                TensorData gate = native_matrix(2, 3, false, 1);
                TensorData up = native_matrix(2, 3, false, 5);
                TensorData down = native_matrix(3, 2, false, 7);
                TensorData gate_bias;
                gate_bias.shape = {2};
                gate_bias.float32_data = {0.375f, -0.625f};
                TensorData up_bias;
                up_bias.shape = {2};
                up_bias.float32_data = {-0.125f, 0.5f};
                TensorData down_bias;
                down_bias.shape = {3};
                down_bias.float32_data = {0.25f, -0.5f, 0.75f};
                if (layout == ExpertLayout::GateUpDown)
                {
                    expert.gate_weight = native_add(model, "gate", native_matrix(2, 3, packed, 1));
                    expert.up_weight = native_add(model, "up", native_matrix(2, 3, packed, 5));
                    expert.gate_bias = native_add(model, "gate_bias", gate_bias);
                    expert.up_bias = native_add(model, "up_bias", up_bias);
                }
                else
                {
                    TensorData combined = native_matrix(4, 3, packed, 0);
                    TensorData bias;
                    bias.shape = {4};
                    bias.float32_data.resize(4);
                    for (uint32_t row = 0; row < 2; ++row)
                    {
                        const uint32_t gate_row = layout == ExpertLayout::PackedGateUpDown ? row : row * 2;
                        const uint32_t up_row = layout == ExpertLayout::PackedGateUpDown ? row + 2 : row * 2 + 1;
                        bias.float32_data[gate_row] = gate_bias.float32_data[row];
                        bias.float32_data[up_row] = up_bias.float32_data[row];
                        if (packed)
                        {
                            const TensorData gate_codes = native_matrix(2, 3, true, 1);
                            const TensorData up_codes = native_matrix(2, 3, true, 5);
                            std::copy_n(gate_codes.quantized_data.data() + row * 2, 2, combined.quantized_data.data() + gate_row * 2);
                            std::copy_n(up_codes.quantized_data.data() + row * 2, 2, combined.quantized_data.data() + up_row * 2);
                        }
                        else
                        {
                            std::copy_n(gate.float32_data.data() + row * 3, 3, combined.float32_data.data() + gate_row * 3);
                            std::copy_n(up.float32_data.data() + row * 3, 3, combined.float32_data.data() + up_row * 3);
                        }
                    }
                    expert.gate_up_weight = native_add(model, "gate_up", std::move(combined));
                    expert.gate_up_bias = native_add(model, "gate_up_bias", std::move(bias));
                }
                expert.down_weight = native_add(model, "down", native_matrix(3, 2, packed, 7));
                expert.down_bias = native_add(model, "down_bias", down_bias);
                ActivationBuffer input(2, 3);
                for (size_t row = 0; row < input.rows(); ++row)
                    for (uint32_t column = 0; column < input.columns(); ++column)
                        input.row(row)[column] = (static_cast<int>(row * 3 + column) - 2) * 0.5f;
                ActivationBuffer actual;
                ExpertWorkspace workspace;
                forward_dense_ffn(model, expert, input, actual, workspace, ExecutionBackend::Cpu);
                for (size_t row = 0; row < input.rows(); ++row)
                {
                    const std::vector<float> token(input.row(row), input.row(row) + input.columns());
                    std::vector<float> gates = dense_project(gate, token);
                    std::vector<float> ups = dense_project(up, token);
                    for (uint32_t column = 0; column < 2; ++column)
                    {
                        float g = gates[column] + gate_bias.float32_data[column];
                        float u = ups[column] + up_bias.float32_data[column];
                        if (activation == ExpertActivation::DeepSeekSwiGlu || activation == ExpertActivation::GptOssSwiGlu)
                        {
                            g = std::min(g, expert.activation_limit);
                            u = std::clamp(u, -expert.activation_limit, expert.activation_limit);
                        }
                        const float alpha = activation == ExpertActivation::GptOssSwiGlu ? 1.702f : 1.0f;
                        gates[column] = g / (1.0f + std::exp(-g * alpha))
                                        * (u + (activation == ExpertActivation::GptOssSwiGlu ? 1.0f : 0.0f));
                    }
                    const std::vector<float> expected = dense_project(down, gates);
                    for (uint32_t column = 0; column < actual.columns(); ++column)
                        dense_near(actual.row(row)[column], expected[column] + down_bias.float32_data[column]);
                }
            }
        }
    }
}

static void test_gated_activation_projection_fallbacks()
{
    for (ExpertActivation activation : {ExpertActivation::GptOssSwiGlu, ExpertActivation::DeepSeekSwiGlu})
    {
        const float limit = activation == ExpertActivation::DeepSeekSwiGlu ? 0.5f : 0.0f;
        const std::array<float, 2> up_values = activation == ExpertActivation::GptOssSwiGlu
                                                   ? std::array<float, 2>{0.25f, -0.5f}
                                                   : std::array<float, 2>{0.5f, 0.25f};
        CompiledModel model;
        model.opt.optimization_flags = 0;
        ExpertPlan expert;
        expert.activation = activation;
        expert.activation_limit = limit;
        TensorData gate;
        gate.shape = {1, 2};
        gate.float32_data = {0.5f, 0.25f};
        expert.gate_weight = native_add(model, "gate", std::move(gate));
        TensorData up;
        up.shape = {1, 2};
        up.float32_data.assign(up_values.begin(), up_values.end());
        expert.up_weight = native_add(model, "up", std::move(up));
        TensorData down;
        down.shape = {2, 1};
        down.float32_data = {2.0f, -1.5f};
        expert.down_weight = native_add(model, "down", std::move(down));
        model.operators.bind_weight_count(model.weights.size());
        expert.vulkan_operator = model.operators.allocate();

        ActivationBuffer input(1, 2);
        input.row(0)[0] = 1.0f;
        input.row(0)[1] = 2.0f;
        const float gate_value = input.row(0)[0] * 0.5f + input.row(0)[1] * 0.25f;
        const float up_value = input.row(0)[0] * up_values[0] + input.row(0)[1] * up_values[1];
        const float activated = activation == ExpertActivation::GptOssSwiGlu
                                    ? scaled_silu(gate_value, 1.702f, 0) * (up_value + 1.0f)
                                    : scaled_silu(std::min(gate_value, limit), 1.0f, 0)
                                          * std::clamp(up_value, -limit, limit);

        ActivationBuffer cpu_output;
        ActivationBuffer vulkan_fallback_output;
        ExpertWorkspace cpu_workspace;
        ExpertWorkspace fallback_workspace;
        forward_dense_ffn(model, expert, input, cpu_output, cpu_workspace, ExecutionBackend::Cpu);
        forward_dense_ffn(model, expert, input, vulkan_fallback_output, fallback_workspace, ExecutionBackend::Vulkan);
        for (uint32_t column = 0; column < 2; ++column)
        {
            const float expected = model.weights.at(expert.down_weight).float32_values()[column] * activated;
            dense_near(cpu_output.row(0)[column], expected);
            dense_near(vulkan_fallback_output.row(0)[column], expected);
        }
    }
}

void test_dense_ffn()
{
    test_gated_activation_projection_fallbacks();
    test_native_expert_layouts();
    for (DType dtype : {DType::Float32, DType::BFloat16})
    {
        const MoeModelDescriptor descriptor = dense_descriptor(dtype);
        dense_check(static_cast<bool>(validate_model_descriptor(descriptor)), "valid descriptor rejected");
        auto memory = plan_model_memory(descriptor, Option{}, 8ull * 1024 * 1024 * 1024);
        dense_check(static_cast<bool>(memory), "memory planning failed");
        const uint64_t ffn_bytes = 3 * 4 * (5 + 3) * (dtype == DType::Float32 ? 4 : 2);
        dense_check(memory.value().estimated_dense_size == 7 * 4 * 2 * 4 + 3 * 4 * 4 + ffn_bytes, "resident FFN byte count");
        dense_check(memory.value().estimated_expert_size == 0 && memory.value().expert_cache_size == 0
                        && memory.value().expert_pair_size == 0 && memory.value().minimum_active_expert_size == 0,
                    "dense model reserves expert storage");
        CompilerOption options;
        options.optimization_flags = 0;
        CompiledModel model;
        auto compiled = compile_model(descriptor, dense_mapping(descriptor), model, HybridMode::CpuOnly, options);
        dense_check(static_cast<bool>(compiled), "compilation failed");
        size_t ffn_nodes = 0;
        for (const ExecutionNode& node : model.graph.nodes)
        {
            if (node.type == ExecutionNodeType::DenseFfn)
                ++ffn_nodes;
            dense_check(node.type != ExecutionNodeType::Router && node.type != ExecutionNodeType::ExpertDispatch
                            && node.type != ExecutionNodeType::ExpertGroup && node.type != ExecutionNodeType::Combine,
                        "dense model has routed graph nodes");
        }
        dense_check(ffn_nodes == descriptor.layers.size(), "missing dense graph nodes");
        const std::array<int32_t, 3> tokens = {1, 3, 5};
        SessionState state;
        SessionStatistics statistics;
        auto prefilled = forward_model(model, tokens, statistics, state, 0);
        dense_check(static_cast<bool>(prefilled), "prefill failed");
        for (size_t row = 0; row < tokens.size(); ++row)
        {
            const std::vector<float> expected = dense_reference(model, tokens[row]);
            for (uint32_t column = 0; column < expected.size(); ++column)
                dense_near(state.logits.row(row)[column], expected[column]);
        }
        dense_no_routes(statistics);
        BatchWorkspace workspace;
        std::array<SessionState, 3> states;
        std::array<SessionStatistics, 3> batch_statistics;
        std::array<std::vector<float>, 3> logits;
        std::array<DecodeBatchEntry, 3> entries;
        for (size_t row = 0; row < entries.size(); ++row)
            entries[row] = {tokens[row], &batch_statistics[row], &states[row], 0, &logits[row]};
        for (uint32_t pass = 0; pass < 2; ++pass)
        {
            auto decoded = forward_decode_batch(model, entries, workspace);
            dense_check(static_cast<bool>(decoded), "staged decode failed");
            for (size_t row = 0; row < entries.size(); ++row)
            {
                dense_check(logits[row].size() == model.descriptor.vocabulary_size, "staged logits shape");
                for (uint32_t column = 0; column < logits[row].size(); ++column)
                    dense_near(logits[row][column], state.logits.row(row)[column]);
                dense_no_routes(batch_statistics[row]);
            }
        }
        auto last = forward_model(model, tokens, statistics, state, 0, LogitsOutput::Last);
        dense_check(static_cast<bool>(last) && state.logits.rows() == 1, "last-only logits failed");
        auto skipped = forward_model(model, tokens, statistics, state, 0, LogitsOutput::None);
        dense_check(static_cast<bool>(skipped) && state.logits.rows() == 0, "skipped logits failed");
    }

    MoeModelDescriptor invalid = dense_descriptor(DType::Float32);
    invalid.layers[0].moe.expert_count = 1;
    dense_check(!validate_model_descriptor(invalid), "dense/MoE overlap admitted");
    invalid = dense_descriptor(DType::Float32);
    invalid.expert_count = invalid.experts_per_token = 1;
    dense_check(!validate_model_descriptor(invalid), "dense model expert globals admitted");
    invalid = dense_descriptor(DType::Float32);
    invalid.layers[0].dense.down_weight_dtype = DType::Int64;
    dense_check(!validate_model_descriptor(invalid), "invalid FFN dtype admitted");
    invalid = dense_descriptor(DType::Float32);
    WeightMapping mapping = dense_mapping(invalid);
    mapping.erase(layer_prefix(0) + "ffn.gate.weight");
    CompiledModel missing;
    dense_check(!compile_model(invalid, std::move(mapping), missing), "missing FFN tensor admitted");
    mapping = dense_mapping(invalid);
    mapping.at(layer_prefix(0) + "ffn.down.weight") = dense_tensor({3, 5}, 0);
    CompiledModel wrong_shape;
    dense_check(!compile_model(invalid, std::move(mapping), wrong_shape), "invalid FFN shape admitted");
}

} // namespace moe
} // namespace ncnn
