#include "test_runtime_decls.h"
#include "test_runtime_support.h"

namespace ncnn {
namespace moe {

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
    CompiledModel graph_driven_model;
    auto graph_driven_status = compile_model(std::move(descriptor).value(),
                                             std::move(mapping).value(), graph_driven_model, HybridMode::CpuOnly);
    if (!graph_driven_status)
    {
        throw std::runtime_error("graph-driven model compilation failed: " + graph_driven_status.error().message);
    }
    check(graph_driven_model.net.layer_plans[0].attention.kind
          == AttentionKind::Standard);
    ActivationBuffer context_hidden(3, 2);
    context_hidden.row(0)[0] = 1.0f;
    context_hidden.row(0)[1] = 0.0f;
    context_hidden.row(1)[0] = 0.0f;
    context_hidden.row(1)[1] = 1.0f;
    context_hidden.row(2)[0] = 0.5f;
    context_hidden.row(2)[1] = -0.5f;
    LayerCache full_cache;
    LayerCache context_cache;
    AttentionScratch full_scratch;
    AttentionScratch context_scratch;
    ActivationBuffer full_output;
    const CompiledModel& compiled = graph_driven_model;
    const AttentionBlockPlan& attention = compiled.net.layer_plans.front().attention;
    const CompiledOperator& query_operator = compiled.operators.at_weight(attention.query_weight);
    check(compiled.operators.size() == compiled.weights.size());
    check(compiled.operators.find_weight(attention.query_weight) != nullptr);
    check(!query_operator.linear);
    check(!support_vulkan_attention(compiled.operators, compiled.net.layer_plans.front().attention));

    // Hybrid models can keep QSA or gated-residual attention on the CPU.
    CompiledModel hybrid_compiled = compiled;
    hybrid_compiled.opt.hybrid_mode = HybridMode::HybridExperts;
    hybrid_compiled.opt.optimization_flags |= OptimizationVulkanAttention;
    auto rebuilt_graph = build_graph(hybrid_compiled, false);
    check(static_cast<bool>(rebuilt_graph));
    const Layer* attention_node = nullptr;
    for (const Layer& node : hybrid_compiled.net.layers)
    {
        if (node.kind == LayerKind::Attention)
        {
            attention_node = &node;
            break;
        }
    }
    check(attention_node != nullptr);
    check(attention_node->backend == ExecutionBackend::Cpu);
    check(attention_node->backend_mask == ExecutionBackendCpu);

    // Speculative layer bindings are compiled as node IDs, so copies and
    // moves must resolve them against their own graph storage.
    CompiledModel speculative_compiled = compiled;
    speculative_compiled.speculative.kind = SpeculativeModelKind::Mtp;
    speculative_compiled.speculative.block_size = 1;
    speculative_compiled.speculative.net.layer_plans = compiled.net.layer_plans;
    auto speculative_graph = build_graph(speculative_compiled, false);
    check(static_cast<bool>(speculative_graph));
    check(speculative_compiled.speculative.layer_nodes.size() == 1);
    const SpeculativeModelPlan::LayerNodes& speculative_nodes = speculative_compiled.speculative.layer_nodes.front();
    check(speculative_nodes.attention != invalid_layer_id);
    check(speculative_nodes.router != invalid_layer_id);
    check(speculative_nodes.expert_dispatch != invalid_layer_id);
    check(speculative_nodes.expert_group != invalid_layer_id);
    check(speculative_nodes.shared_expert_group == invalid_layer_id);
    check(speculative_nodes.combine != invalid_layer_id);
    check(speculative_compiled.speculative.net.layers[speculative_nodes.attention].kind
          == LayerKind::Attention);
    check(speculative_compiled.speculative.net.layers[speculative_nodes.router].kind
          == LayerKind::Router);
    check(speculative_compiled.speculative.net.layers[speculative_nodes.expert_dispatch].kind
          == LayerKind::ExpertDispatch);
    check(speculative_compiled.speculative.net.layers[speculative_nodes.expert_group].kind
          == LayerKind::ExpertGroup);
    check(speculative_compiled.speculative.net.layers[speculative_nodes.combine].kind
          == LayerKind::Combine);

    CompiledModel copied_speculative = speculative_compiled;
    const SpeculativeModelPlan::LayerNodes& copied_nodes = copied_speculative.speculative.layer_nodes.front();
    check(&copied_speculative.speculative.net.layers[copied_nodes.attention]
          != &speculative_compiled.speculative.net.layers[speculative_nodes.attention]);
    check(copied_speculative.speculative.net.layers[copied_nodes.attention].kind
          == LayerKind::Attention);

    CompiledModel missing_attention_speculative = speculative_compiled;
    missing_attention_speculative.speculative.net.layer_plans.front().attention.kind = AttentionKind::None;
    auto missing_attention_graph = build_graph(missing_attention_speculative,
                                               false);
    check(!missing_attention_graph);
    check(missing_attention_graph.error().code == ErrorCode::InternalError);
    check(missing_attention_graph.error().message
          == "speculative graph is missing a required layer node");

    CompiledModel moved_speculative = std::move(copied_speculative);
    const SpeculativeModelPlan::LayerNodes& moved_nodes = moved_speculative.speculative.layer_nodes.front();
    check(moved_speculative.speculative.net.layers[moved_nodes.expert_group].kind
          == LayerKind::ExpertGroup);

    Net rescheduled_speculative_graph = moved_speculative.speculative.net;
    auto rescheduled_speculative_status = schedule_graph(rescheduled_speculative_graph,
                                                         ExecutionBackendCpu);
    check(static_cast<bool>(rescheduled_speculative_status));
    check(rescheduled_speculative_graph.order == moved_speculative.speculative.net.order);
    check(rescheduled_speculative_graph.layers[moved_nodes.attention].kind
          == LayerKind::Attention);
    check(rescheduled_speculative_graph.layers[moved_nodes.expert_group].kind
          == LayerKind::ExpertGroup);

    moved_speculative.speculative.layer_nodes.clear();
    auto rebuilt_speculative = build_graph(moved_speculative, false);
    check(static_cast<bool>(rebuilt_speculative));
    check(moved_speculative.speculative.layer_nodes.size() == 1);
    const SpeculativeModelPlan::LayerNodes& rebuilt_nodes = moved_speculative.speculative.layer_nodes.front();
    check(rebuilt_nodes.attention != invalid_layer_id);
    check(moved_speculative.speculative.net.layers[rebuilt_nodes.attention].kind
          == LayerKind::Attention);

    auto full_attention = forward_attention(compiled.weights,
                                            compiled.operators,
                                            compiled.net.layer_plans.front().attention,
                                            ExecutionBackend::Cpu,
                                            7,
                                            full_cache,
                                            full_scratch,
                                            context_hidden,
                                            full_output,
                                            g_test_optimization_flags);
    check(static_cast<bool>(full_attention));
    auto appended_context = append_attention_context(compiled.weights,
                                                     compiled.operators,
                                                     compiled.net.layer_plans.front().attention,
                                                     ExecutionBackend::Cpu,
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

    ActivationBuffer long_hidden(80, 2);
    for (size_t row = 0; row < long_hidden.rows(); ++row)
    {
        long_hidden.row(row)[0] = static_cast<float>(static_cast<int>(row % 13) - 6) * 0.0625f;
        long_hidden.row(row)[1] = static_cast<float>(static_cast<int>((row * 3) % 17) - 8) * 0.03125f;
    }
    const uint64_t attention_reference_flags = g_test_optimization_flags
                                               & ~OptimizationCpuFlashAttention
                                               & ~OptimizationCpuSplitKvAttention;
    LayerCache reference_long_cache;
    LayerCache flash_long_cache;
    AttentionScratch reference_long_scratch;
    AttentionScratch flash_long_scratch;
    ActivationBuffer reference_long_output;
    ActivationBuffer flash_long_output;
    auto reference_long = forward_attention(compiled.weights,
                                            compiled.operators,
                                            compiled.net.layer_plans.front().attention,
                                            ExecutionBackend::Cpu,
                                            0,
                                            reference_long_cache,
                                            reference_long_scratch,
                                            long_hidden,
                                            reference_long_output,
                                            attention_reference_flags);
    auto flash_long = forward_attention(compiled.weights,
                                        compiled.operators,
                                        compiled.net.layer_plans.front().attention,
                                        ExecutionBackend::Cpu,
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

    ActivationBuffer split_prefix(512, 2);
    for (size_t row = 0; row < split_prefix.rows(); ++row)
    {
        split_prefix.row(row)[0] = static_cast<float>(static_cast<int>(row % 19) - 9) * 0.015625f;
        split_prefix.row(row)[1] = static_cast<float>(static_cast<int>((row * 5) % 23) - 11) * 0.015625f;
    }
    LayerCache reference_split_cache;
    LayerCache split_kv_cache;
    AttentionScratch reference_split_scratch;
    AttentionScratch split_kv_scratch;
    auto reference_context = append_attention_context(compiled.weights,
                                                      compiled.operators,
                                                      compiled.net.layer_plans.front().attention,
                                                      ExecutionBackend::Cpu,
                                                      0,
                                                      reference_split_cache,
                                                      reference_split_scratch,
                                                      split_prefix,
                                                      attention_reference_flags);
    auto split_context = append_attention_context(compiled.weights,
                                                  compiled.operators,
                                                  compiled.net.layer_plans.front().attention,
                                                  ExecutionBackend::Cpu,
                                                  0,
                                                  split_kv_cache,
                                                  split_kv_scratch,
                                                  split_prefix,
                                                  g_test_optimization_flags);
    check(static_cast<bool>(reference_context));
    check(static_cast<bool>(split_context));
    ActivationBuffer split_token(1, 2);
    split_token.row(0)[0] = 0.125f;
    split_token.row(0)[1] = -0.0625f;
    ActivationBuffer reference_split_output;
    ActivationBuffer split_kv_output;
    auto reference_decode = forward_attention(compiled.weights,
                                              compiled.operators,
                                              compiled.net.layer_plans.front().attention,
                                              ExecutionBackend::Cpu,
                                              512,
                                              reference_split_cache,
                                              reference_split_scratch,
                                              split_token,
                                              reference_split_output,
                                              attention_reference_flags);
    auto split_decode = forward_attention(compiled.weights,
                                          compiled.operators,
                                          compiled.net.layer_plans.front().attention,
                                          ExecutionBackend::Cpu,
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

    AttentionBlockPlan sliding_plan = compiled.net.layer_plans.front().attention;
    sliding_plan.sliding_window = 2;
    std::array<LayerCache, 1> sliding_context_cache;
    check(static_cast<bool>(begin_state_cache_transaction(sliding_context_cache,
                                                          context_hidden.rows())));
    AttentionScratch sliding_context_scratch;
    auto rejected_sliding_context = append_attention_context(compiled.weights,
                                                             compiled.operators,
                                                             sliding_plan,
                                                             ExecutionBackend::Cpu,
                                                             7,
                                                             sliding_context_cache.front(),
                                                             sliding_context_scratch,
                                                             context_hidden,
                                                             g_test_optimization_flags);
    check(static_cast<bool>(!rejected_sliding_context));
    check(static_cast<bool>(rejected_sliding_context.error().code
                            == ErrorCode::UnsupportedModel));
    check(sliding_context_cache.front().token_count == 0);
    check(static_cast<bool>(finish_state_cache_transaction(sliding_context_cache,
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
    for (const Layer& node : cpu_compiled.net.layers)
        check(static_cast<bool>(node.backend == ExecutionBackend::Cpu));

    auto cpu_session = runtime.create_session(cpu_model.value());
    check(static_cast<bool>(cpu_session));
    const std::vector<int32_t> prompt = {0, 1, 0};
    auto cpu_prefill = cpu_session.value()->prefill(prompt);
    check(static_cast<bool>(cpu_prefill));

    if (has_usable_vulkan_device(runtime.info(), RuntimeVulkanAttention))
    {
        Option hybrid_options;
        hybrid_options.hybrid_mode = HybridMode::HybridExperts;
        auto hybrid_model = runtime.load_model(package.path(), hybrid_options);
        check(static_cast<bool>(hybrid_model));
        const Net& hybrid_graph = model_compiled(*hybrid_model.value()).net;
        size_t vulkan_attention_nodes = 0;
        size_t vulkan_lm_head_nodes = 0;
        for (const Layer& node : hybrid_graph.layers)
        {
            if (node.kind == LayerKind::Attention && node.backend == ExecutionBackend::Vulkan)
            {
                ++vulkan_attention_nodes;
            }
            if (node.kind == LayerKind::LmHead && node.backend == ExecutionBackend::Vulkan)
            {
                ++vulkan_lm_head_nodes;
            }
            if (node.kind == LayerKind::ExpertGroup)
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
    struct ActivationCase
    {
        ExpertActivation activation;
        float limit;
        uint64_t optimization_flags;
    };
    const uint64_t fast_silu_flags = g_test_optimization_flags
                                     | static_cast<uint64_t>(OptimizationCpuFastSilu);
    const uint64_t scalar_silu_flags = g_test_optimization_flags
                                       & ~static_cast<uint64_t>(OptimizationCpuFastSilu);
    const std::array<ActivationCase, 4> separate_activation_cases = {{{ExpertActivation::Silu, 0.0f, fast_silu_flags},
                                                                      {ExpertActivation::DeepSeekSwiGlu, 0.0f, fast_silu_flags},
                                                                      {ExpertActivation::Silu, 0.0f, scalar_silu_flags},
                                                                      {ExpertActivation::DeepSeekSwiGlu, 0.125f, fast_silu_flags}}};
    const std::array<ActivationCase, 5> packed_activation_cases = {{{ExpertActivation::Silu, 0.0f, scalar_silu_flags},
                                                                    {ExpertActivation::Silu, 0.0f, fast_silu_flags},
                                                                    {ExpertActivation::DeepSeekSwiGlu, 0.0f, scalar_silu_flags},
                                                                    {ExpertActivation::DeepSeekSwiGlu, 0.0f, fast_silu_flags},
                                                                    {ExpertActivation::DeepSeekSwiGlu, 0.125f, g_test_optimization_flags}}};
    // Gate/up activation must preserve each row's source stride and tail.
    for (ExpertLayout layout : {ExpertLayout::PackedGateUpDown,
                                ExpertLayout::InterleavedGateUpDown,
                                ExpertLayout::GateUpDown})
    {
        for (DType dtype : {DType::Float32, DType::BFloat16})
        {
            for (bool use_bias : {false, true})
            {
                constexpr uint32_t hidden_size = 3;
                // 17 columns gives ARM, AVX2 and AVX512 full vectors plus a one-value tail.
                constexpr uint32_t intermediate_size = 17;
                const bool separate_weights = layout == ExpertLayout::GateUpDown;
                if (separate_weights && use_bias)
                    continue;
                CompiledModel model;
                MoeBlockPlan moe;
                moe.has_shared_expert = true;
                ExpertPlan& expert = moe.shared_expert;
                expert.layout = layout;
                expert.activation = layout == ExpertLayout::PackedGateUpDown
                                        ? ExpertActivation::DeepSeekSwiGlu
                                        : ExpertActivation::GptOssSwiGlu;
                expert.activation_limit = 0.125f;
                auto add_weight = [&](const char* name,
                                      uint32_t rows,
                                      uint32_t columns,
                                      uint32_t phase = 0) {
                    TensorData tensor;
                    tensor.dtype = dtype;
                    tensor.shape = {rows, columns};
                    for (size_t index = 0; index < static_cast<size_t>(rows) * columns; ++index)
                    {
                        const float value = static_cast<float>(static_cast<int>((index * 7 + 3 + phase) % 19) - 9)
                                            * 0.0625f;
                        if (dtype == DType::Float32)
                            tensor.float32_data.push_back(value);
                        else
                            tensor.bfloat16_data.push_back(float_to_bfloat16(value));
                    }
                    auto ret = model.weights.add(name, std::move(tensor));
                    check(static_cast<bool>(ret));
                    return ret.value();
                };
                if (separate_weights)
                {
                    expert.gate_weight = add_weight("gate", intermediate_size, hidden_size);
                    expert.up_weight = add_weight("up", intermediate_size, hidden_size, 5);
                }
                else
                {
                    expert.gate_up_weight = add_weight("gate_up", intermediate_size * 2, hidden_size);
                }
                expert.down_weight = add_weight("down", hidden_size, intermediate_size);
                if (use_bias)
                {
                    TensorData bias;
                    bias.dtype = DType::Float32;
                    bias.shape = {intermediate_size * 2};
                    bias.float32_data.assign(intermediate_size * 2, 0.03125f);
                    auto ret = model.weights.add("gate_up_bias", std::move(bias));
                    check(static_cast<bool>(ret));
                    expert.gate_up_bias = ret.value();
                }
                const TensorData& down = model.weights.at(expert.down_weight);
                const TensorData* gate_up = separate_weights ? nullptr : &model.weights.at(expert.gate_up_weight);
                const TensorData* gate_weight = separate_weights
                                                    ? &model.weights.at(expert.gate_weight)
                                                    : nullptr;
                const TensorData* up_weight = separate_weights
                                                  ? &model.weights.at(expert.up_weight)
                                                  : nullptr;
                ActivationBuffer output(4, hidden_size);
                ExpertWorkspace workspace;
                const std::byte* output_storage = output.bytes().data();
                const std::byte* workspace_projection_storage = nullptr;
                const std::byte* workspace_gate_storage = nullptr;
                uint64_t workspace_projection_capacity = 0;
                uint64_t workspace_gate_capacity = 0;
                const bool packed_weights = layout == ExpertLayout::PackedGateUpDown;
                const size_t activation_case_count = separate_weights
                                                         ? separate_activation_cases.size()
                                                     : packed_weights
                                                         ? packed_activation_cases.size()
                                                         : 1;
                for (size_t activation_case_index = 0;
                     activation_case_index < activation_case_count;
                     ++activation_case_index)
                {
                    if (separate_weights)
                    {
                        expert.activation = separate_activation_cases[activation_case_index].activation;
                        expert.activation_limit = separate_activation_cases[activation_case_index].limit;
                        model.opt.optimization_flags = separate_activation_cases[activation_case_index].optimization_flags;
                    }
                    else if (packed_weights)
                    {
                        expert.activation = packed_activation_cases[activation_case_index].activation;
                        expert.activation_limit = packed_activation_cases[activation_case_index].limit;
                        model.opt.optimization_flags = packed_activation_cases[activation_case_index].optimization_flags;
                    }
                    else
                    {
                        expert.activation_limit = 0.125f;
                        model.opt.optimization_flags = g_test_optimization_flags;
                    }
                    for (size_t rows : {size_t{3}, size_t{1}, size_t{3}})
                    {
                        ActivationBuffer input(rows, hidden_size);
                        for (size_t row = 0; row < rows; ++row)
                            for (uint32_t column = 0; column < hidden_size; ++column)
                                input.row(row)[column] = static_cast<float>(static_cast<int>(row * 3 + column) - 4) * 0.25f;
                        ActivationBuffer projected;
                        ActivationBuffer projected_gate;
                        ActivationBuffer projected_up;
                        if (separate_weights)
                        {
                            projected_gate = forward_linear(*gate_weight, input, model.opt.optimization_flags);
                            projected_up = forward_linear(*up_weight, input, model.opt.optimization_flags);
                        }
                        else
                        {
                            projected = use_bias
                                            ? forward_linear(*gate_up, model.weights.at(expert.gate_up_bias), input, model.opt.optimization_flags)
                                            : forward_linear(*gate_up, input, model.opt.optimization_flags);
                        }
                        ActivationBuffer activated(rows, intermediate_size);
                        for (size_t row = 0; row < rows; ++row)
                        {
                            for (uint32_t column = 0; column < intermediate_size; ++column)
                            {
                                if (separate_weights)
                                {
                                    float gate = projected_gate.row(row)[column];
                                    float up = projected_up.row(row)[column];
                                    if (expert.activation == ExpertActivation::DeepSeekSwiGlu
                                        && expert.activation_limit > 0.0f)
                                    {
                                        gate = std::min(gate, expert.activation_limit);
                                        up = std::clamp(up, -expert.activation_limit, expert.activation_limit);
                                    }
                                    activated.row(row)[column] = scaled_silu(gate,
                                                                             1.0f,
                                                                             model.opt.optimization_flags)
                                                                 * up;
                                }
                                else
                                {
                                    float gate = projected.row(row)[packed_weights ? column : column * 2];
                                    float up = projected.row(row)[packed_weights ? intermediate_size + column : column * 2 + 1];
                                    if (packed_weights)
                                    {
                                        if (expert.activation == ExpertActivation::DeepSeekSwiGlu
                                            && expert.activation_limit > 0.0f)
                                        {
                                            gate = std::min(gate, expert.activation_limit);
                                            up = std::clamp(up, -expert.activation_limit, expert.activation_limit);
                                        }
                                        activated.row(row)[column] = scaled_silu(gate,
                                                                                 1.0f,
                                                                                 model.opt.optimization_flags)
                                                                     * up;
                                    }
                                    else
                                    {
                                        gate = std::min(gate, expert.activation_limit);
                                        up = std::clamp(up, -expert.activation_limit, expert.activation_limit);
                                        activated.row(row)[column] = scaled_silu(gate,
                                                                                 1.702f,
                                                                                 model.opt.optimization_flags)
                                                                     * (up + 1.0f);
                                    }
                                }
                            }
                        }
                        const ActivationBuffer expected = forward_linear(down, activated, model.opt.optimization_flags);
                        forward_shared_expert(model, moe, input, output, workspace);
                        check(output.rows() == rows && output.columns() == hidden_size);
                        check(output.bytes().data() == output_storage);
                        const bool fast_packed_vector_case = packed_weights
                                                             && expert.activation_limit <= 0.0f
                                                             && has_flag(model.opt.optimization_flags, OptimizationCpuFastSilu);
                        if (packed_weights
                            && !has_flag(model.opt.optimization_flags, OptimizationCpuFastSilu))
                        {
                            check(workspace.projection.values().size() == activated.values().size());
                            for (size_t index = 0; index < activated.values().size(); ++index)
                                check(workspace.projection.values()[index] == activated.values()[index]);
                        }
                        for (size_t index = 0; index < expected.values().size(); ++index)
                            check_near(output.values()[index],
                                       expected.values()[index],
                                       fast_packed_vector_case ? 1e-5f : 1e-6f);
                        if (workspace_projection_storage == nullptr)
                        {
                            workspace_projection_storage = workspace.projection.bytes().data();
                            workspace_gate_storage = workspace.gate.bytes().data();
                            workspace_projection_capacity = workspace.projection.allocated_bytes();
                            workspace_gate_capacity = workspace.gate.allocated_bytes();
                        }
                        else
                        {
                            check(workspace.projection.bytes().data() == workspace_projection_storage);
                            check(workspace.gate.bytes().data() == workspace_gate_storage);
                            check(workspace.projection.allocated_bytes() >= workspace_projection_capacity);
                            check(workspace.gate.allocated_bytes() >= workspace_gate_capacity);
                        }
                    }
                }
            }
        }
    }

    TemporaryModelPackage package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));
    const CompiledModel& compiled = model_compiled(*model.value());
    check(compiled.descriptor.layers.front().moe.shared_expert_count == 0);
    check(!compiled.net.layer_plans.front().moe.has_shared_expert);
    MoeModelDescriptor baseline_descriptor = compiled.descriptor;
    check(static_cast<bool>(validate_model_descriptor(baseline_descriptor)));

    MoeModelDescriptor shared_descriptor = baseline_descriptor;
    shared_descriptor.layers.front().moe.shared_expert_count = 1;
    shared_descriptor.layers.front().moe.shared_expert_weight_dtype = DType::Float32;
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
    CompiledModel shared_model;
    auto shared_status = compile_model(shared_descriptor, std::move(mapping).value(), shared_model);
    check(static_cast<bool>(shared_status));
    shared_model.opt.optimization_flags = g_test_optimization_flags;
    const MoeBlockPlan& shared_plan = shared_model.net.layer_plans.front().moe;
    check(shared_plan.has_shared_expert);
    check(shared_plan.shared_expert.gate_weight != invalid_tensor_handle);
    check(shared_plan.shared_expert.up_weight != invalid_tensor_handle);
    check(shared_plan.shared_expert.down_weight != invalid_tensor_handle);

    ActivationBuffer shared_input;
    ActivationBuffer shared_output(8, shared_descriptor.hidden_size);
    const std::byte* output_storage = shared_output.bytes().data();
    const uint64_t output_capacity = shared_output.allocated_bytes();
    ExpertWorkspace shared_workspace;
    const std::byte* shared_workspace_projection_storage = nullptr;
    const std::byte* shared_workspace_gate_storage = nullptr;
    uint64_t shared_workspace_projection_capacity = 0;
    uint64_t shared_workspace_gate_capacity = 0;
    for (size_t rows : {size_t{3}, size_t{1}, size_t{3}})
    {
        shared_input.reset(rows, shared_descriptor.hidden_size, false);
        for (size_t row = 0; row < rows; ++row)
            std::fill_n(shared_input.row(row), shared_input.columns(), 1.0f);
        // A reused destination must be overwritten, including after a shrink.
        for (size_t row = 0; row < shared_output.rows(); ++row)
            std::fill_n(shared_output.row(row), shared_output.columns(), std::numeric_limits<float>::quiet_NaN());
        forward_shared_expert(shared_model,
                              shared_plan,
                              shared_input,
                              shared_output,
                              shared_workspace);
        check(shared_output.rows() == rows);
        check(shared_output.columns() == shared_descriptor.hidden_size);
        check(shared_output.bytes().data() == output_storage);
        check(shared_output.allocated_bytes() == output_capacity);
        for (float value : shared_output.values())
            check(value == 0.0f);
        if (shared_workspace_projection_storage == nullptr)
        {
            shared_workspace_projection_storage = shared_workspace.projection.bytes().data();
            shared_workspace_gate_storage = shared_workspace.gate.bytes().data();
            shared_workspace_projection_capacity = shared_workspace.projection.allocated_bytes();
            shared_workspace_gate_capacity = shared_workspace.gate.allocated_bytes();
        }
        else
        {
            check(shared_workspace.projection.bytes().data() == shared_workspace_projection_storage);
            check(shared_workspace.gate.bytes().data() == shared_workspace_gate_storage);
            check(shared_workspace.projection.allocated_bytes() >= shared_workspace_projection_capacity);
            check(shared_workspace.gate.allocated_bytes() >= shared_workspace_gate_capacity);
        }
    }

    auto nonzero_mapping = adapter.map_weights(fixture, compiled.descriptor);
    check(static_cast<bool>(nonzero_mapping));
    for (const std::string& projection : {"gate", "up", "down"})
    {
        TensorData weight;
        weight.dtype = DType::Float32;
        weight.shape = projection == "down"
                           ? std::vector<uint32_t>{shared_descriptor.hidden_size, shared_descriptor.intermediate_size}
                           : std::vector<uint32_t>{shared_descriptor.intermediate_size, shared_descriptor.hidden_size};
        weight.float32_data = projection == "gate"
                                  ? std::vector<float>{0.5f, -0.25f, 0.75f, 0.25f}
                              : projection == "up"
                                  ? std::vector<float>{1.0f, 0.25f, -0.5f, 1.5f}
                                  : std::vector<float>{0.75f, -0.25f, 0.5f, 1.25f};
        nonzero_mapping.value().emplace("layers.0.shared_expert." + projection + ".weight", std::move(weight));
    }
    CompiledModel nonzero_shared_model;
    auto nonzero_shared_status = compile_model(shared_descriptor, std::move(nonzero_mapping).value(), nonzero_shared_model);
    check(static_cast<bool>(nonzero_shared_status));
    nonzero_shared_model.opt.optimization_flags = g_test_optimization_flags;
    // Keep the previous routed-then-shared schedule as an independent
    // numerical reference for both ordinary and staged execution.
    const auto restore_serial_shared_schedule = [](Net& graph) {
        Layer* dispatch = nullptr;
        Layer* routed = nullptr;
        Layer* shared = nullptr;
        for (Layer& node : graph.layers)
        {
            if (node.kind == LayerKind::ExpertDispatch)
                dispatch = &node;
            else if (node.kind == LayerKind::ExpertGroup)
                routed = &node;
            else if (node.kind == LayerKind::SharedExpertGroup)
                shared = &node;
        }
        check(dispatch != nullptr && routed != nullptr && shared != nullptr);
        check(has_flag(dispatch->flags, LayerRequestExperts));
        check(std::find(graph.order.begin(), graph.order.end(), shared->id)
              < std::find(graph.order.begin(), graph.order.end(), routed->id));
        dispatch->flags &= ~LayerRequestExperts;
        const auto retarget_control = [&](LayerId from, LayerId to, LayerId new_from, LayerId new_to) {
            for (Blob& tensor : graph.blobs)
            {
                if (tensor.lifetime == BlobLifetime::Control && tensor.producer == from
                    && tensor.consumers.size() == 1 && tensor.consumers.front() == to)
                {
                    tensor.producer = new_from;
                    tensor.consumers.front() = new_to;
                    return true;
                }
            }
            return false;
        };
        check(retarget_control(dispatch->id, shared->id, routed->id, shared->id));
        check(retarget_control(shared->id, routed->id, dispatch->id, routed->id));
        check(static_cast<bool>(schedule_graph(graph, ExecutionBackendCpu)));
    };
    CompiledModel serial_shared_model = nonzero_shared_model;
    restore_serial_shared_schedule(serial_shared_model.net);
    SessionState reordered_state;
    SessionState serial_state;
    SessionStatistics reordered_statistics;
    SessionStatistics serial_statistics;
    for (const std::vector<int32_t>& tokens : {std::vector<int32_t>{0, 1, 2}, std::vector<int32_t>{3}})
    {
        const uint64_t position = tokens.size() == 1 ? 3 : 0;
        auto actual_status = forward_model(nonzero_shared_model, tokens, reordered_statistics, reordered_state, position);
        auto expected_status = forward_model(serial_shared_model, tokens, serial_statistics, serial_state, position);
        check(static_cast<bool>(actual_status) && static_cast<bool>(expected_status));
        const auto actual = batch_to_vectors(reordered_state.logits);
        const auto expected = batch_to_vectors(serial_state.logits);
        check(actual == expected);
        check(reordered_statistics.expert_assignments == serial_statistics.expert_assignments);
        check(reordered_statistics.expert_batches == serial_statistics.expert_batches);
    }

    // The speculative executor uses compiled node bindings rather than the
    // main execution loop. It must honor the same shared-expert ordering.
    CompiledModel mtp_shared_model = nonzero_shared_model;
    SpeculativeModelPlan& mtp = mtp_shared_model.speculative;
    mtp.kind = SpeculativeModelKind::Mtp;
    mtp.block_size = 2;
    mtp.net.layer_plans = mtp_shared_model.net.layer_plans;
    AttentionBlockPlan& mtp_attention = mtp.net.layer_plans.front().attention;
    mtp_attention.kind = AttentionKind::Standard;
    mtp_attention.head_count = 1;
    mtp_attention.kv_head_count = 1;
    mtp_attention.head_dimension = 2;
    mtp_attention.value_head_dimension = 2;
    mtp_attention.pre_attention_norm_weight = mtp_shared_model.final_norm_weight;
    mtp_attention.query_weight = mtp.net.layer_plans.front().moe.shared_expert.up_weight;
    mtp_attention.key_weight = mtp_attention.query_weight;
    mtp_attention.value_weight = mtp_attention.query_weight;
    mtp_attention.output_weight = mtp_attention.query_weight;
    mtp.mtp_embedding_norm_weight = mtp_shared_model.final_norm_weight;
    mtp.mtp_hidden_norm_weight = mtp_shared_model.final_norm_weight;
    mtp.final_norm_weight = mtp_shared_model.final_norm_weight;
    TensorData mtp_projection;
    mtp_projection.dtype = DType::Float32;
    mtp_projection.shape = {2, 4};
    mtp_projection.float32_data = {1.0f, 0.0f, 0.25f, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f};
    auto projection_handle = mtp_shared_model.weights.add("test.mtp.projection", std::move(mtp_projection));
    check(static_cast<bool>(projection_handle));
    mtp.mtp_input_projection_weight = projection_handle.value();
    auto mtp_graph = build_graph(mtp_shared_model, false);
    if (!mtp_graph)
        throw std::runtime_error("shared MTP graph failed: " + mtp_graph.error().message);
    CompiledModel serial_mtp_model = mtp_shared_model;
    restore_serial_shared_schedule(serial_mtp_model.speculative.net);
    std::array<SessionState, 2> mtp_states;
    std::array<SessionStatistics, 2> mtp_statistics;
    for (size_t index = 0; index < mtp_states.size(); ++index)
    {
        mtp_states[index].speculative_layers.resize(1);
        mtp_states[index].mtp_pending_target_hidden.reset(1, 2, false);
        mtp_states[index].mtp_pending_target_hidden.row(0)[0] = 1.0f;
        mtp_states[index].mtp_pending_target_hidden.row(0)[1] = 0.25f;
    }
    const auto pending_bytes = mtp_states[0].mtp_pending_target_hidden.bytes();
    const std::byte* pending_data = pending_bytes.data();
    const std::vector<std::byte> pending_snapshot(pending_bytes.begin(), pending_bytes.end());
    SessionStatistics failure_statistics = mtp_statistics[0];
    check(static_cast<bool>(begin_state_cache_transaction(mtp_states[0].speculative_layers,
                                                          mtp.block_size)));
    size_t sampler_calls = 0;
    const SpeculativeSampler failing_sampler = [&sampler_calls](std::span<const float>) -> Result<int32_t> {
        if (sampler_calls++ == 0)
            return 0;
        return Error{ErrorCode::InternalError, "test sampler failure"};
    };
    SpeculativeProposal mtp_actual;
    auto failed_mtp = propose_speculative(mtp_shared_model,
                                          1,
                                          failure_statistics,
                                          mtp_states[0],
                                          1,
                                          mtp_actual,
                                          failing_sampler);
    auto rolled_back_mtp_cache = finish_state_cache_transaction(mtp_states[0].speculative_layers,
                                                                0);
    check(static_cast<bool>(rolled_back_mtp_cache));
    check(!failed_mtp);
    check(failed_mtp.error().code == ErrorCode::InternalError);
    check(failed_mtp.error().message == "test sampler failure");
    check(sampler_calls == 2);
    check(mtp_actual.token_ids.size() == 1 && mtp_actual.confidence_logits.size() == 1);
    check(mtp_actual.logits.size() == mtp.block_size);
    check(mtp_states[0].mtp_pending_target_hidden.rows() == 1);
    check(mtp_states[0].mtp_pending_target_hidden.columns() == 2);
    check(mtp_states[0].mtp_pending_target_hidden.bytes().data() == pending_data);
    check(mtp_states[0].mtp_pending_target_hidden.bytes().size() == pending_snapshot.size());
    check(std::equal(mtp_states[0].mtp_pending_target_hidden.bytes().begin(),
                     mtp_states[0].mtp_pending_target_hidden.bytes().end(),
                     pending_snapshot.begin(),
                     pending_snapshot.end()));
    std::vector<std::span<const float>> sampled_mtp_logits;
    const SpeculativeSampler first_token = [&sampled_mtp_logits](std::span<const float> logits) -> Result<int32_t> {
        sampled_mtp_logits.push_back(logits);
        return 0;
    };
    auto mtp_status = propose_speculative(mtp_shared_model, 1, mtp_statistics[0], mtp_states[0], 1, mtp_actual, first_token);
    check(static_cast<bool>(mtp_status));
    check(sampled_mtp_logits.size() == mtp_actual.logits.size());
    for (size_t row = 0; row < sampled_mtp_logits.size(); ++row)
    {
        const std::span<const float> retained = mtp_actual.logits[row].values();
        check(sampled_mtp_logits[row].data() == retained.data());
        check(sampled_mtp_logits[row].size() == retained.size());
        check(std::equal(sampled_mtp_logits[row].begin(), sampled_mtp_logits[row].end(), retained.begin()));
    }
    sampled_mtp_logits.clear();
    SpeculativeProposal mtp_expected;
    auto expected_status = propose_speculative(serial_mtp_model, 1, mtp_statistics[1], mtp_states[1], 1, mtp_expected, first_token);
    check(static_cast<bool>(expected_status));
    check(sampled_mtp_logits.size() == mtp_expected.logits.size());
    check(mtp_actual.logits.size() == mtp_expected.logits.size());
    for (size_t row = 0; row < sampled_mtp_logits.size(); ++row)
    {
        const std::span<const float> retained = mtp_expected.logits[row].values();
        const std::span<const float> actual = mtp_actual.logits[row].values();
        check(sampled_mtp_logits[row].data() == retained.data());
        check(sampled_mtp_logits[row].size() == retained.size());
        check(std::equal(sampled_mtp_logits[row].begin(), sampled_mtp_logits[row].end(), retained.begin()));
        check(retained.size() == actual.size());
        check(std::equal(retained.begin(), retained.end(), actual.begin()));
    }
    check(mtp_actual.token_ids == mtp_expected.token_ids);
    check(mtp_statistics[0].expert_assignments == mtp_statistics[1].expert_assignments);

    SessionState repeated_mtp_state;
    repeated_mtp_state.speculative_layers.resize(1);
    repeated_mtp_state.mtp_pending_target_hidden.reset(1, 2, false);
    repeated_mtp_state.mtp_pending_target_hidden.row(0)[0] = 1.0f;
    repeated_mtp_state.mtp_pending_target_hidden.row(0)[1] = 0.25f;
    SessionStatistics repeated_mtp_statistics;
    const std::array<const float*, 2> mtp_storage = {
        mtp_actual.logits[0].values().data(), mtp_actual.logits[1].values().data()};
    const auto* mtp_tokens = mtp_actual.token_ids.data();
    const auto* mtp_confidence = mtp_actual.confidence_logits.data();
    sampled_mtp_logits.clear();
    auto repeated_mtp = propose_speculative(mtp_shared_model, 1, repeated_mtp_statistics,
                                            repeated_mtp_state, 1, mtp_actual, first_token);
    check(static_cast<bool>(repeated_mtp));
    check(mtp_actual.token_ids == mtp_expected.token_ids);
    check(mtp_actual.confidence_logits == mtp_expected.confidence_logits);
    check(mtp_actual.token_ids.data() == mtp_tokens);
    check(mtp_actual.confidence_logits.data() == mtp_confidence);
    check(sampled_mtp_logits.size() == mtp_actual.logits.size());
    for (size_t row = 0; row < mtp_storage.size(); ++row)
    {
        check(mtp_actual.logits[row].values().data() == mtp_storage[row]);
        check(sampled_mtp_logits[row].data() == mtp_storage[row]);
        check(std::equal(mtp_actual.logits[row].values().begin(), mtp_actual.logits[row].values().end(),
                         mtp_expected.logits[row].values().begin(), mtp_expected.logits[row].values().end()));
    }

    const MoeBlockPlan& nonzero_shared_plan = nonzero_shared_model.net.layer_plans.front().moe;
    ActivationBuffer nonzero_shared_input(2, shared_descriptor.hidden_size);
    nonzero_shared_input.row(0)[0] = 1.0f;
    nonzero_shared_input.row(0)[1] = 0.0f;
    nonzero_shared_input.row(1)[0] = 0.0f;
    nonzero_shared_input.row(1)[1] = 1.0f;
    ActivationBuffer nonzero_shared_output;
    ExpertWorkspace nonzero_shared_workspace;
    forward_shared_expert(nonzero_shared_model,
                          nonzero_shared_plan,
                          nonzero_shared_input,
                          nonzero_shared_output,
                          nonzero_shared_workspace);
    check(std::any_of(nonzero_shared_output.values().begin(),
                      nonzero_shared_output.values().end(),
                      [](float value) { return value != 0.0f; }));
    check(nonzero_shared_output.row(0)[0] != nonzero_shared_output.row(1)[0]
          || nonzero_shared_output.row(0)[1] != nonzero_shared_output.row(1)[1]);

    std::array<SessionState, 3> staged_states;
    std::array<SessionStatistics, 3> staged_statistics;
    std::array<std::vector<float>, 3> staged_logits;
    BatchWorkspace shared_batch_workspace;
    check(!has_flag(nonzero_shared_model.opt.flags, OptionRankAdaptivePrefetch));
    auto check_staged_batch = [&](std::span<const DecodeBatchEntry> entries) {
        std::array<uint64_t, 3> expert_batches_before{};
        std::array<std::array<uint64_t, maximum_expert_route_ranks>, 3> rank_demands_before{};
        std::array<std::array<uint64_t, maximum_expert_route_ranks>, 3> queue_times_before{};
        for (size_t session_index = 0; session_index < entries.size(); ++session_index)
        {
            const SessionStatistics& statistics = *entries[session_index].statistics;
            expert_batches_before[session_index] = statistics.expert_batches;
            rank_demands_before[session_index] = statistics.expert_route_rank_demands;
            queue_times_before[session_index] = statistics.expert_route_rank_demand_queue_time_microseconds;
        }
        auto staged_status = forward_decode_batch(nonzero_shared_model, entries, shared_batch_workspace);
        check(static_cast<bool>(staged_status));
        for (size_t session_index = 0; session_index < entries.size(); ++session_index)
        {
            check(entries[session_index].logits != nullptr);
            SessionState reference_state;
            SessionStatistics reference_statistics;
            const std::array<int32_t, 1> input = {entries[session_index].input_id};
            auto expected_status = forward_model(serial_shared_model, input,
                                                 reference_statistics, reference_state, 0);
            check(static_cast<bool>(expected_status));
            check(reference_state.logits.rows() == 1);
            check(entries[session_index].logits->size() == reference_state.logits.columns());
            for (size_t column = 0; column < reference_state.logits.columns(); ++column)
                check_near((*entries[session_index].logits)[column], reference_state.logits.row(0)[column], 1e-5f);
            const SessionStatistics& actual_statistics = *entries[session_index].statistics;
            check(actual_statistics.expert_batches - expert_batches_before[session_index]
                  == reference_statistics.expert_batches);
            for (size_t rank = 0; rank < maximum_expert_route_ranks; ++rank)
            {
                check(actual_statistics.expert_route_rank_demands[rank]
                          - rank_demands_before[session_index][rank]
                      == reference_statistics.expert_route_rank_demands[rank]);
                check(actual_statistics.expert_route_rank_demand_queue_time_microseconds[rank]
                      == queue_times_before[session_index][rank]);
            }
            check(entries[session_index].state->execution_state.shared_expert_output.rows() == 0);
        }
    };
    const std::array<DecodeBatchEntry, 3> first_staged_entries = {{
        {0, &staged_statistics[0], &staged_states[0], 0, &staged_logits[0]},
        {1, &staged_statistics[1], &staged_states[1], 0, &staged_logits[1]},
        {2, &staged_statistics[2], &staged_states[2], 0, &staged_logits[2]},
    }};
    check_staged_batch(first_staged_entries);

    const std::array<DecodeBatchEntry, 2> second_staged_entries = {{
        {3, &staged_statistics[0], &staged_states[0], 0, &staged_logits[0]},
        {0, &staged_statistics[2], &staged_states[2], 0, &staged_logits[2]},
    }};
    check_staged_batch(second_staged_entries);

    const std::array<DecodeBatchEntry, 3> third_staged_entries = {{
        {1, &staged_statistics[0], &staged_states[0], 0, &staged_logits[0]},
        {3, &staged_statistics[1], &staged_states[1], 0, &staged_logits[1]},
        {0, &staged_statistics[2], &staged_states[2], 0, &staged_logits[2]},
    }};
    check_staged_batch(third_staged_entries);

    auto missing_shared_mapping = adapter.map_weights(fixture, compiled.descriptor);
    check(static_cast<bool>(missing_shared_mapping));
    CompiledModel missing_shared_model;
    auto missing_shared_status = compile_model(shared_descriptor,
                                               std::move(missing_shared_mapping).value(),
                                               missing_shared_model);
    check(!missing_shared_status);
    check(missing_shared_status.error().code == ErrorCode::InvalidModel);
    check(missing_shared_status.error().message
          == "missing tensor: layers.0.shared_expert.gate.weight");

    MoeModelDescriptor gated_descriptor = baseline_descriptor;
    gated_descriptor.layers.front().moe.flags |= MoeDescriptorSharedExpertGate;
    auto gated_mapping = adapter.map_weights(fixture, compiled.descriptor);
    check(static_cast<bool>(gated_mapping));
    CompiledModel gated_model;
    auto gated_status = compile_model(gated_descriptor, std::move(gated_mapping).value(), gated_model);
    check(!gated_status);
    check(gated_status.error().code == ErrorCode::InvalidModel);
    check(gated_status.error().message == "shared Expert gate requires a shared Expert");

    shared_descriptor.layers.front().moe.shared_expert_count = 2;
    check(static_cast<bool>(validate_model_descriptor(shared_descriptor)));
    auto multiple_mapping = adapter.map_weights(fixture, compiled.descriptor);
    check(static_cast<bool>(multiple_mapping));
    CompiledModel multiple_model;
    auto multiple_status = compile_model(shared_descriptor, std::move(multiple_mapping).value(), multiple_model);
    check(!multiple_status);
    check(multiple_status.error().code == ErrorCode::UnsupportedModel);
    check(multiple_status.error().message == "the CPU runtime supports one shared Expert per MoE block");
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
    inconsistent_moe_descriptor.layers.front().moe.expert_count++;
    auto inconsistent_moe_status = validate_model_descriptor(inconsistent_moe_descriptor);
    check(!inconsistent_moe_status);
    check(inconsistent_moe_status.error().code == ErrorCode::InvalidModel);
    check(inconsistent_moe_status.error().message == "layer MoE dimensions do not match the model descriptor");
    MoeModelDescriptor invalid_top_k_descriptor = baseline_descriptor;
    invalid_top_k_descriptor.layers.front().moe.top_k = invalid_top_k_descriptor.layers.front().moe.expert_count + 1;
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
    auto baseline_memory = plan_model_memory(baseline_descriptor, options, UINT64_C(8) * 1024 * 1024 * 1024);
    auto inactive_memory = plan_model_memory(inactive_attention_descriptor, options, UINT64_C(8) * 1024 * 1024 * 1024);
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
    known_moe_flags_descriptor.layers.front().moe.shared_expert_count = 1;
    known_moe_flags_descriptor.layers.front().moe.flags = MoeDescriptorRouterBias
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
    unknown_moe_flags_descriptor.layers.front().moe.flags = std::numeric_limits<uint32_t>::max();
    auto unknown_moe_flags_status = validate_model_descriptor(unknown_moe_flags_descriptor);
    check(!unknown_moe_flags_status);
    check(unknown_moe_flags_status.error().code == ErrorCode::InvalidModel);
    check(unknown_moe_flags_status.error().message
          == "model descriptor layer has unknown MoE flags");

    MoeModelDescriptor unknown_kind_descriptor = baseline_descriptor;
    unknown_kind_descriptor.layers.front().attention.kind = static_cast<AttentionKind>(-1);
    auto unknown_kind_status = validate_model_descriptor(unknown_kind_descriptor);
    check(!unknown_kind_status);
    check(unknown_kind_status.error().code == ErrorCode::InvalidModel);
    check(unknown_kind_status.error().message == "model descriptor layer has an invalid attention kind");

    MoeModelDescriptor invalid_normalization_descriptor = baseline_descriptor;
    invalid_normalization_descriptor.layers.front().moe.normalization = static_cast<RouterNormalization>(-1);
    auto invalid_normalization_status = validate_model_descriptor(invalid_normalization_descriptor);
    check(!invalid_normalization_status);
    check(invalid_normalization_status.error().code == ErrorCode::InvalidModel);
    check(invalid_normalization_status.error().message == "model descriptor layer has an invalid router normalization");

    const auto check_invalid_descriptor = [](const MoeModelDescriptor& invalid_descriptor, const char* message) {
        auto invalid_status = validate_model_descriptor(invalid_descriptor);
        check(!invalid_status);
        check(invalid_status.error().code == ErrorCode::InvalidModel);
        check(invalid_status.error().message == message);
        CompiledModel invalid_model;
        auto invalid_compile_status = compile_model(invalid_descriptor, WeightMapping{}, invalid_model);
        check(!invalid_compile_status);
        check(invalid_compile_status.error().code == ErrorCode::InvalidModel);
        check(invalid_compile_status.error().message == message);
    };
    constexpr uint32_t gate_up_projection_limit = std::numeric_limits<uint32_t>::max() / 2;
    constexpr uint32_t gate_up_projection_overflow = gate_up_projection_limit + 1;
    constexpr const char* gate_up_projection_error = "gate-up projection dimensions overflow";
    for (ExpertLayout layout : {
             ExpertLayout::PackedGateUpDown,
             ExpertLayout::InterleavedGateUpDown})
    {
        for (uint32_t intermediate_size : {
                 gate_up_projection_limit,
                 gate_up_projection_overflow})
        {
            MoeModelDescriptor descriptor = baseline_descriptor;
            descriptor.intermediate_size = intermediate_size;
            for (LayerDescriptor& layer : descriptor.layers)
            {
                layer.moe.intermediate_size = intermediate_size;
                layer.moe.layout = layout;
            }
            if (intermediate_size == gate_up_projection_limit)
                check(static_cast<bool>(validate_model_descriptor(descriptor)));
            else
                check_invalid_descriptor(descriptor, gate_up_projection_error);
        }
    }
    for (ExpertLayout layout : {ExpertLayout::UpDown, ExpertLayout::GateUpDown})
    {
        MoeModelDescriptor descriptor = baseline_descriptor;
        descriptor.intermediate_size = gate_up_projection_overflow;
        for (LayerDescriptor& layer : descriptor.layers)
        {
            layer.moe.intermediate_size = gate_up_projection_overflow;
            layer.moe.layout = layout;
        }
        check(static_cast<bool>(validate_model_descriptor(descriptor)));
    }
    MoeModelDescriptor projection_bias_descriptor = baseline_descriptor;
    projection_bias_descriptor.intermediate_size = gate_up_projection_overflow;
    for (LayerDescriptor& layer : projection_bias_descriptor.layers)
    {
        layer.moe.intermediate_size = gate_up_projection_overflow;
        layer.moe.layout = ExpertLayout::GateUpDown;
        layer.moe.flags |= MoeDescriptorProjectionBias;
    }
    check_invalid_descriptor(projection_bias_descriptor, gate_up_projection_error);
    for (SpeculativeModelKind kind : {SpeculativeModelKind::Mtp, SpeculativeModelKind::DSpark})
    {
        MoeModelDescriptor descriptor = baseline_descriptor;
        descriptor.intermediate_size = gate_up_projection_overflow;
        descriptor.speculative_kind = kind;
        descriptor.speculative_layer_count = 1;
        descriptor.speculative_target_layer_ids = kind == SpeculativeModelKind::DSpark
                                                      ? std::vector<uint32_t>{0}
                                                      : std::vector<uint32_t>{};
        for (LayerDescriptor& layer : descriptor.layers)
        {
            layer.moe.intermediate_size = gate_up_projection_overflow;
            layer.moe.layout = ExpertLayout::UpDown;
            layer.moe.flags &= ~MoeDescriptorProjectionBias;
        }
        check_invalid_descriptor(descriptor, gate_up_projection_error);
    }
    for (float value : {
             0.0f,
             -1.0f,
             std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity()})
    {
        MoeModelDescriptor invalid_norm_descriptor = baseline_descriptor;
        invalid_norm_descriptor.norm_epsilon = value;
        check_invalid_descriptor(invalid_norm_descriptor, "norm_epsilon must be finite and positive");

        MoeModelDescriptor invalid_scaling_descriptor = baseline_descriptor;
        invalid_scaling_descriptor.layers.front().moe.routed_scaling_factor = value;
        check_invalid_descriptor(invalid_scaling_descriptor, "routed scaling factor must be finite and positive");
    }
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
    check_invalid_descriptor(duplicate_target_descriptor,
                             "speculative target layer IDs contain duplicates");

    MoeModelDescriptor out_of_range_target_descriptor = target_order_descriptor;
    out_of_range_target_descriptor.speculative_target_layer_ids = {2, 0, 3};
    check_invalid_descriptor(out_of_range_target_descriptor,
                             "speculative target layer ID is out of range");

    MoeModelDescriptor zero_count_kind_descriptor = baseline_descriptor;
    zero_count_kind_descriptor.speculative_kind = SpeculativeModelKind::DSpark;
    check_invalid_descriptor(zero_count_kind_descriptor,
                             "speculative fields require speculative layers");

    MoeModelDescriptor zero_count_targets_descriptor = baseline_descriptor;
    zero_count_targets_descriptor.speculative_target_layer_ids = {0};
    check_invalid_descriptor(zero_count_targets_descriptor,
                             "speculative fields require speculative layers");

    MoeModelDescriptor missing_speculative_kind_descriptor = target_order_descriptor;
    missing_speculative_kind_descriptor.speculative_kind = SpeculativeModelKind::None;
    missing_speculative_kind_descriptor.speculative_layer_count = 1;
    missing_speculative_kind_descriptor.speculative_target_layer_ids = {0};
    check_invalid_descriptor(missing_speculative_kind_descriptor,
                             "speculative layer kind is not configured");

    MoeModelDescriptor invalid_enum_descriptor = baseline_descriptor;
    invalid_enum_descriptor.layers.front().moe.score_function = static_cast<RouterScoreFunction>(-1);
    check_invalid_descriptor(invalid_enum_descriptor,
                             "model descriptor layer has an invalid router score function");
    invalid_enum_descriptor = baseline_descriptor;
    invalid_enum_descriptor.layers.front().moe.activation = static_cast<ExpertActivation>(-1);
    check_invalid_descriptor(invalid_enum_descriptor,
                             "model descriptor layer has an invalid expert activation");
    invalid_enum_descriptor = baseline_descriptor;
    invalid_enum_descriptor.layers.front().moe.layout = static_cast<ExpertLayout>(-1);
    check_invalid_descriptor(invalid_enum_descriptor,
                             "model descriptor layer has an invalid expert layout");

    for (RouterScoreFunction score_function : {
             RouterScoreFunction::Softmax,
             RouterScoreFunction::Sigmoid,
             RouterScoreFunction::SqrtSoftplus})
    {
        MoeModelDescriptor valid_descriptor = baseline_descriptor;
        valid_descriptor.layers.front().moe.score_function = score_function;
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
        valid_descriptor.layers.front().moe.activation = activation;
        check(static_cast<bool>(validate_model_descriptor(valid_descriptor)));
    }
    for (ExpertLayout layout : {
             ExpertLayout::UpDown,
             ExpertLayout::GateUpDown,
             ExpertLayout::PackedGateUpDown,
             ExpertLayout::InterleavedGateUpDown})
    {
        MoeModelDescriptor valid_descriptor = baseline_descriptor;
        valid_descriptor.layers.front().moe.layout = layout;
        check(static_cast<bool>(validate_model_descriptor(valid_descriptor)));
    }
    invalid_enum_descriptor = baseline_descriptor;
    invalid_enum_descriptor.hyper_connection_kind = static_cast<HyperConnectionKind>(-1);
    check_invalid_descriptor(invalid_enum_descriptor,
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

    CompiledModel invalid_standard_flags_model;
    auto invalid_standard_flags_compile_status = compile_model(invalid_standard_flags_descriptor, WeightMapping{}, invalid_standard_flags_model);
    check(!invalid_standard_flags_compile_status);
    check(invalid_standard_flags_compile_status.error().code == ErrorCode::InvalidModel);
    check(invalid_standard_flags_compile_status.error().message
          == "model descriptor layer has incompatible attention flags");
    CompiledModel invalid_moe_flags_model;
    auto invalid_moe_flags_status = compile_model(unknown_moe_flags_descriptor, WeightMapping{}, invalid_moe_flags_model);
    check(!invalid_moe_flags_status);
    check(invalid_moe_flags_status.error().code == ErrorCode::InvalidModel);
    check(invalid_moe_flags_status.error().message
          == "model descriptor layer has unknown MoE flags");
    CompiledModel invalid_normalization_model;
    auto invalid_normalization_compile_status = compile_model(invalid_normalization_descriptor, WeightMapping{}, invalid_normalization_model);
    check(!invalid_normalization_compile_status);
    check(invalid_normalization_compile_status.error().code == ErrorCode::InvalidModel);
    check(invalid_normalization_compile_status.error().message == "model descriptor layer has an invalid router normalization");
    const Net& graph = compiled.net;
    check(compiled.net.layer_plans.front().attention.kind == AttentionKind::None);
    check(static_cast<bool>(graph.layers.size() == 7));
    check(static_cast<bool>(!graph.blobs.empty()));
    const std::vector<LayerKind> expected_types = {
        LayerKind::TokenEmbedding,
        LayerKind::Router,
        LayerKind::ExpertDispatch,
        LayerKind::ExpertGroup,
        LayerKind::Combine,
        LayerKind::FinalNorm,
        LayerKind::LmHead,
    };
    for (size_t index = 0; index < expected_types.size(); ++index)
    {
        check(static_cast<bool>(graph.layers[index].id == index));
        check(static_cast<bool>(graph.layers[index].kind == expected_types[index]));
        check(static_cast<bool>(graph.layers[index].backend == ExecutionBackend::Cpu));
    }
    check(static_cast<bool>(graph.layers[3].flags == 0));
    check(static_cast<bool>(graph.layers[3].plan_index == 0));
    const auto assignments = std::find_if(graph.blobs.begin(), graph.blobs.end(), [](const Blob& tensor) {
        return tensor.name.find("expert_dispatch.assignments") != std::string::npos;
    });
    check(assignments != graph.blobs.end());
    check(assignments->lifetime == BlobLifetime::Control);
    check(assignments->producer == 2);
    check(assignments->consumers == std::vector<LayerId>{3});
    const auto experts_output = std::find_if(graph.blobs.begin(), graph.blobs.end(), [](const Blob& tensor) {
        return tensor.name.find("experts.output") != std::string::npos;
    });
    check(experts_output != graph.blobs.end());
    check(experts_output->producer == 3);
    check(std::find(experts_output->consumers.begin(), experts_output->consumers.end(), LayerId{4})
          != experts_output->consumers.end());

    check(static_cast<bool>(graph.validate_order()));
    check(static_cast<bool>(graph.order.size() == graph.layers.size()));
    for (LayerId node_id : graph.order)
        check(static_cast<bool>(graph.layers[node_id].backend == ExecutionBackend::Cpu));

    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));
    const std::array<int32_t, 1> prompt = {0};
    auto prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill));
    const SessionStatistics session_statistics = session.value()->statistics();
    check(static_cast<bool>(session_statistics.expert_batch_weight_bytes > 0));
    check(static_cast<bool>(session_statistics.expert_route_weight_bytes >= session_statistics.expert_batch_weight_bytes));
    Net rescheduled_graph = graph;
    auto rescheduled_status = schedule_graph(rescheduled_graph, ExecutionBackendCpu);
    check(static_cast<bool>(rescheduled_status));
    check(static_cast<bool>(rescheduled_graph.order == graph.order));
    auto repeated_rescheduled_status = schedule_graph(rescheduled_graph, ExecutionBackendCpu);
    check(static_cast<bool>(repeated_rescheduled_status));
    check(static_cast<bool>(rescheduled_graph.order == graph.order));

    for (LayerKind type : {LayerKind::ExpertGroup, LayerKind::SharedExpertGroup})
    {
        for (uint32_t binding : {invalid_plan_index, static_cast<uint32_t>(graph.layer_plans.size())})
        {
            Net invalid_graph = graph;
            invalid_graph.layers[3].kind = type;
            invalid_graph.layers[3].plan_index = binding;
            auto ret = schedule_graph(invalid_graph, ExecutionBackendCpu);
            check(!ret && ret.error().code == ErrorCode::InvalidModel);
            check(ret.error().message == (binding == invalid_plan_index ? "execution graph node has an invalid layer plan binding" : "execution graph node layer plan is out of range"));
        }
    }

    const auto add_test_edge = [](Net& graph, LayerId producer, LayerId consumer) {
        Blob tensor;
        tensor.id = static_cast<BlobId>(graph.blobs.size());
        tensor.name = "test.edge." + std::to_string(tensor.id);
        tensor.dtype = DType::Int32;
        tensor.shape = {1};
        tensor.lifetime = BlobLifetime::Control;
        tensor.producer = producer;
        tensor.consumers = {consumer};
        graph.blobs.push_back(std::move(tensor));
    };

    Net branch_graph;
    for (LayerId node_id = 0; node_id < 5; ++node_id)
    {
        Layer node;
        node.id = node_id;
        node.kind = LayerKind::FinalNorm;
        node.backend = ExecutionBackend::Cpu;
        node.backend_mask = ExecutionBackendCpu;
        node.weights = {node_id};
        branch_graph.layers.push_back(std::move(node));
    }
    add_test_edge(branch_graph, 3, 0);
    add_test_edge(branch_graph, 1, 2);
    add_test_edge(branch_graph, 0, 4);
    add_test_edge(branch_graph, 2, 4);
    auto branch_status = schedule_graph(branch_graph, ExecutionBackendCpu);
    check(static_cast<bool>(branch_status));
    check(static_cast<bool>(branch_graph.order == std::vector<LayerId>{1, 3, 2, 0, 4}));

    Net duplicate_dependency_graph = branch_graph;
    duplicate_dependency_graph.blobs.back().consumers.push_back(duplicate_dependency_graph.blobs.back().consumers.front());
    auto duplicate_schedule_status = schedule_graph(duplicate_dependency_graph, ExecutionBackendCpu);
    check(!duplicate_schedule_status);
    check(duplicate_schedule_status.error().message == "execution tensor contains a duplicate consumer");

    Net cpu_only_graph = graph;
    cpu_only_graph.layers.back().backend = ExecutionBackend::Vulkan;
    cpu_only_graph.layers.back().backend_mask |= ExecutionBackendVulkan;
    auto cpu_only_status = schedule_graph(cpu_only_graph, ExecutionBackendCpu);
    check(static_cast<bool>(cpu_only_status));
    check(static_cast<bool>(cpu_only_graph.layers.back().backend == ExecutionBackend::Cpu));
    check(static_cast<bool>(cpu_only_graph.order.size() == cpu_only_graph.layers.size()));

    Net hybrid_graph = graph;
    hybrid_graph.layers.back().backend_mask |= ExecutionBackendVulkan;
    constexpr uint32_t scheduling_backends = ExecutionBackendCpu | ExecutionBackendVulkan;
    auto hybrid_status = schedule_graph(hybrid_graph, scheduling_backends);
    check(static_cast<bool>(hybrid_status));
    check(static_cast<bool>(hybrid_graph.validate_order()));
    check(static_cast<bool>(hybrid_graph.order.size() == hybrid_graph.layers.size()));
    check(static_cast<bool>(hybrid_graph.layers.back().backend == ExecutionBackend::Vulkan));
    for (size_t index = 0; index + 1 < hybrid_graph.order.size(); ++index)
        check(static_cast<bool>(hybrid_graph.layers[hybrid_graph.order[index]].backend == ExecutionBackend::Cpu));
    check(static_cast<bool>(hybrid_graph.layers[hybrid_graph.order.back()].backend == ExecutionBackend::Vulkan));

    for (LayerKind type : {
             LayerKind::Attention,
             LayerKind::ExpertGroup,
             LayerKind::SharedExpertGroup,
             LayerKind::DenseFfn})
    {
        Net eligible_graph = graph;
        Layer& eligible_node = eligible_graph.layers[3];
        eligible_node.kind = type;
        eligible_node.backend = ExecutionBackend::Cpu;
        eligible_node.backend_mask |= ExecutionBackendVulkan;
        eligible_graph.layers.front().backend = ExecutionBackend::Vulkan;

        auto cpu_status = schedule_graph(eligible_graph, ExecutionBackendCpu);
        check(static_cast<bool>(cpu_status));
        check(static_cast<bool>(eligible_node.backend == ExecutionBackend::Cpu));
        for (const Layer& node : eligible_graph.layers)
        {
            if (node.backend_mask == ExecutionBackendCpu)
                check(static_cast<bool>(node.backend == ExecutionBackend::Cpu));
        }

        eligible_graph.layers.front().backend = ExecutionBackend::Vulkan;
        auto hybrid_status = schedule_graph(eligible_graph, scheduling_backends);
        check(static_cast<bool>(hybrid_status));
        check(static_cast<bool>(eligible_node.backend == ExecutionBackend::Vulkan));
        for (const Layer& node : eligible_graph.layers)
        {
            if (node.backend_mask == ExecutionBackendCpu)
                check(static_cast<bool>(node.backend == ExecutionBackend::Cpu));
        }

        auto rescheduled_cpu_status = schedule_graph(eligible_graph, ExecutionBackendCpu);
        check(static_cast<bool>(rescheduled_cpu_status));
        check(static_cast<bool>(eligible_node.backend == ExecutionBackend::Cpu));
    }

    Net vulkan_only_graph;
    Layer vulkan_dense_node;
    vulkan_dense_node.id = 0;
    vulkan_dense_node.kind = LayerKind::LmHead;
    vulkan_dense_node.backend = ExecutionBackend::Cpu;
    vulkan_dense_node.backend_mask = scheduling_backends;
    vulkan_dense_node.weights = {0};
    vulkan_only_graph.layers.push_back(std::move(vulkan_dense_node));
    auto vulkan_only_status = schedule_graph(vulkan_only_graph, ExecutionBackendVulkan);
    check(static_cast<bool>(vulkan_only_status));
    check(static_cast<bool>(vulkan_only_graph.layers.front().backend == ExecutionBackend::Vulkan));

    auto no_backend_status = schedule_graph(vulkan_only_graph, 0);
    check(!no_backend_status);
    check(no_backend_status.error().code == ErrorCode::InvalidArgument);
    check(no_backend_status.error().message == "runtime scheduler requires at least one backend");

    Net invalid_order_net = hybrid_graph;
    const LayerId delayed_token_node = invalid_order_net.order.front();
    invalid_order_net.order.erase(invalid_order_net.order.begin());
    invalid_order_net.order.push_back(delayed_token_node);
    auto invalid_order_status = invalid_order_net.validate_order();
    check(!invalid_order_status);
    check(static_cast<bool>(invalid_order_status.error().code == ErrorCode::InvalidModel));
    check(static_cast<bool>(invalid_order_status.error().message == "execution schedule violates a tensor dependency"));

    Net invalid_backend_graph = hybrid_graph;
    invalid_backend_graph.layers.back().backend_mask = ExecutionBackendCpu;
    auto invalid_backend_status = invalid_backend_graph.validate_order();
    check(!invalid_backend_status);
    check(static_cast<bool>(invalid_backend_status.error().code == ErrorCode::InvalidModel));
    check(static_cast<bool>(invalid_backend_status.error().message == "execution schedule node backend is not allowed by its backend mask"));

    Net invalid_dependency_graph = hybrid_graph;
    invalid_dependency_graph.blobs.back().consumers.push_back(invalid_layer_id);
    auto invalid_dependency_status = schedule_graph(invalid_dependency_graph, scheduling_backends);
    check(!invalid_dependency_status);
    check(static_cast<bool>(invalid_dependency_status.error().code == ErrorCode::InvalidModel));
    check(static_cast<bool>(invalid_dependency_status.error().message == "execution tensor consumer is out of range"));

    for (LayerId invalid_id : {static_cast<LayerId>(graph.layers.size()), invalid_layer_id, LayerId{0}})
    {
        Net invalid_graph = hybrid_graph;
        invalid_graph.layers.back().id = invalid_id;
        auto invalid_schedule_status = schedule_graph(invalid_graph, scheduling_backends);
        check(!invalid_schedule_status);
        check(invalid_schedule_status.error().code == ErrorCode::InvalidModel);
        check(invalid_schedule_status.error().message == "execution graph node ids must be contiguous and index-aligned");
    }

    Net unavailable_graph = hybrid_graph;
    unavailable_graph.layers.back().id = invalid_layer_id;
    unavailable_graph.layers.front().backend_mask = 0;
    auto unavailable_schedule_status = schedule_graph(unavailable_graph, scheduling_backends);
    check(!unavailable_schedule_status);
    check(unavailable_schedule_status.error().code == ErrorCode::UnsupportedModel);
    check(unavailable_schedule_status.error().message == "execution node has no available backend: TokenEmbedding #0");

    Net cyclic;
    Layer cyclic_router;
    cyclic_router.id = 0;
    cyclic_router.kind = LayerKind::Router;
    cyclic_router.backend = ExecutionBackend::Cpu;
    cyclic_router.backend_mask = ExecutionBackendCpu;
    cyclic_router.plan_index = 0;
    Layer cyclic_combine;
    cyclic_combine.id = 1;
    cyclic_combine.kind = LayerKind::Combine;
    cyclic_combine.backend = ExecutionBackend::Cpu;
    cyclic_combine.backend_mask = ExecutionBackendCpu;
    cyclic_combine.plan_index = 0;
    cyclic.layers = {
        std::move(cyclic_router),
        std::move(cyclic_combine),
    };
    cyclic.layer_plans.resize(1);
    add_test_edge(cyclic, 0, 1);
    add_test_edge(cyclic, 1, 0);
    auto invalid_schedule_status = schedule_graph(cyclic, ExecutionBackendCpu);
    check(static_cast<bool>(!invalid_schedule_status));
    check(static_cast<bool>(invalid_schedule_status.error().code == ErrorCode::InvalidModel));
    check(invalid_schedule_status.error().message == "execution graph contains a dependency cycle");
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
    auto nested_value = read_manifest_uint32(json, "value");
    check(!nested_value);
    check(nested_value.error().message == "manifest is missing integer field: value");
    auto kind = read_manifest_string(json, "kind");
    check(static_cast<bool>(kind));
    check(kind.value() == "second");
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
    check_near(nested_scale.value(), 8.5f, 1e-6f);
    auto nested_flag = read_manifest_bool(json, "nested_flag");
    check(static_cast<bool>(nested_flag));
    check(nested_flag.value());
    auto flag = read_manifest_bool(json, "flag");
    check(static_cast<bool>(flag));
    check(flag.value());

    auto nested_object = read_manifest_object(json, "nested");
    check(static_cast<bool>(nested_object));
    auto nested_kind = read_manifest_string(nested_object.value(), "kind");
    check(static_cast<bool>(nested_kind));
    check(nested_kind.value() == "first");
    auto nested_object_value = read_manifest_uint32(nested_object.value(), "value");
    check(static_cast<bool>(nested_object_value));
    check(nested_object_value.value() == 7);
    auto nested_object_scale = read_manifest_float(nested_object.value(), "nested_scale");
    check(static_cast<bool>(nested_object_scale));
    check_near(nested_object_scale.value(), 3.25f, 1e-6f);
    auto nested_object_flag = read_manifest_bool(nested_object.value(), "nested_flag");
    check(static_cast<bool>(nested_object_flag));
    check(!nested_object_flag.value());

    auto escaped = read_manifest_string(R"({"text":"line\n\u4f60\u597d\"\\\/"})", "text");
    check(static_cast<bool>(escaped));
    check(escaped.value() == std::string("line\n") + "\xE4\xBD\xA0\xE5\xA5\xBD\"\\/");
    auto invalid_escape = read_manifest_string(R"({"text":"bad\q"})", "text");
    check(!invalid_escape);
    auto invalid_string_suffix = read_manifest_string(R"({"text":"x"junk})", "text");
    check(!invalid_string_suffix);
    auto escaped_key = read_manifest_uint32(R"({"\u0076alue":7})", "value");
    check(static_cast<bool>(escaped_key));
    check(escaped_key.value() == 7);
    auto surrogate_pair = read_manifest_string(R"({"text":"\uD83D\uDE00"})", "text");
    check(static_cast<bool>(surrogate_pair));
    check(surrogate_pair.value() == "\xF0\x9F\x98\x80");
    auto isolated_high_surrogate = read_manifest_string(R"({"text":"\uD83D"})", "text");
    check(!isolated_high_surrogate);
    auto isolated_low_surrogate = read_manifest_string(R"({"text":"\uDE00"})", "text");
    check(!isolated_low_surrogate);

    const std::string utf8_text = "\xc2\xa2\xe4\xbd\xa0\xe5\xa5\xbd\xf0\x9f\x98\x80";
    auto utf8_string = read_manifest_string(std::string(R"({"text":")") + utf8_text + R"("})", "text");
    check(static_cast<bool>(utf8_string));
    check(utf8_string.value() == utf8_text);
    for (const char* text : {"null", "true", "false", "-0", "1.25e-2", "[]", "{}",
                             R"({"values":[null,true,false,-1,0.5,{},["text"]]})"})
    {
        size_t position = 0;
        check(parse_json_value(text, position));
        check(position == std::string_view(text).size());
    }
    for (const char* text : {"foo", "01", "-", ".5", "1.", "1e+", "truefalse", "[]garbage",
                             "[1 2]", "[1,]", "[,1]", "[1,,2]", "[true:false]", "[}",
                             R"({foo:1})", R"({"x" 1})", R"({"x":})", R"({"x":foo})",
                             R"({"x":1,})", R"({"x":1 "y":2})", R"({"x":[]garbage})"})
    {
        size_t position = 0;
        check(!parse_json_value(text, position));
    }
    size_t json_position = 0;
    check(parse_json_value(" [1] , 2", json_position));
    check(json_position == 4);
    // Overflow nesting must not recurse or impose a new protocol limit.
    const std::string deep_json = std::string(4096, '[') + "null" + std::string(4096, ']');
    json_position = 0;
    check(parse_json_value(deep_json, json_position));
    check(json_position == deep_json.size());
    json_position = 0;
    check(!parse_json_value(std::string_view(deep_json).substr(0, deep_json.size() - 1), json_position));
    for (const char* bytes : {"\x80", "\xc0\xaf", "\xc2", "\xe4\xbd", "\xe0\x80\xaf",
                              "\xed\xa0\x80", "\xf0\x80\x80\xaf", "\xf4\x90\x80\x80",
                              "\xf5\x80\x80\x80", "\xc4\xe3\xba\xc3"})
    {
        const std::string text = std::string("\"") + bytes + "\"";
        check(!read_manifest_string(std::string(R"({"text":)") + text + "}", "text"));
        // Parsing without decoding must enforce the same boundary.
        size_t position = 0;
        check(!parse_json_string(text, position, nullptr));
        position = 0;
        check(!parse_json_value(std::string(R"({"text":)") + text + "}", position));
        position = 0;
        check(!parse_json_value(std::string(R"({"text":1,)") + bytes + "}", position));
    }

    auto balanced_object = read_manifest_object(R"({"object":{"value":7,"nested":{"flag":true}}})", "object");
    check(static_cast<bool>(balanced_object));
    auto owned_nested_object = [] {
        const std::string source = R"({"object":{"text":"first\nsecond","value":7}})";
        return read_manifest_object(source, "object");
    }();
    check(static_cast<bool>(owned_nested_object));
    auto owned_nested_text = read_manifest_string(owned_nested_object.value(), "text");
    check(static_cast<bool>(owned_nested_text));
    check(owned_nested_text.value() == "first\nsecond");
    auto invalid_object_type = read_manifest_object(R"({"object":[1]})", "object", "Test ");
    check(!invalid_object_type);
    check(invalid_object_type.error().message == "invalid Test object field: object");
    auto invalid_object_balance = read_manifest_object(R"({"object":{"value":[1}}]})", "object");
    check(!invalid_object_balance);
    auto missing_root_object = read_manifest_object(R"({"object":{"value":1})", "object");
    check(!missing_root_object);

    for (const char* scalar : {"-1", "1.0", "1 2", "truejunk", "\"1\""})
    {
        auto invalid = read_manifest_uint32(std::string(R"({"value":)") + scalar + "}", "value");
        check(!invalid);
        check(invalid.error().code == ErrorCode::InvalidModel);
    }
    for (const char* scalar : {"1.2junk", "1 2", "truejunk", "\"1\""})
    {
        auto invalid = read_manifest_float(std::string(R"({"value":)") + scalar + "}", "value");
        check(!invalid);
        check(invalid.error().code == ErrorCode::InvalidModel);
    }
    for (const char* scalar : {"truejunk", "false 1", "\"true\""})
    {
        auto invalid = read_manifest_bool(std::string(R"({"value":)") + scalar + "}", "value");
        check(!invalid);
        check(invalid.error().code == ErrorCode::InvalidModel);
    }

    auto integers = read_manifest_uint32_array(R"({"values":[0, 2, 4294967295]})", "values");
    check(static_cast<bool>(integers));
    check(integers.value() == std::vector<uint32_t>{0, 2, std::numeric_limits<uint32_t>::max()});
    for (const char* array : {"[-1]", "[2.5]", "[,2]", "[2,]", "[2,,3]", "[2,junk]"})
    {
        auto invalid = read_manifest_uint32_array(std::string(R"({"values":)") + array + "}", "values");
        check(!invalid);
        check(invalid.error().code == ErrorCode::InvalidModel);
    }
    auto array_uint32_overflow = read_manifest_uint32_array(R"({"values":[4294967296]})", "values");
    check(!array_uint32_overflow);
    check(array_uint32_overflow.error().message == "integer array value is out of range: values");
    auto array_uint64_overflow = read_manifest_uint32_array(R"({"values":[18446744073709551616]})", "values");
    check(!array_uint64_overflow);
    check(array_uint64_overflow.error().message == "invalid integer array: values");
    for (const char* array : {"[01]", "[+1]"})
    {
        auto invalid = read_manifest_uint32_array(std::string(R"({"values":)") + array + "}", "values");
        check(!invalid);
        check(invalid.error().message == "manifest is missing integer array: values");
    }
    auto strings = read_manifest_string_array(R"({"values":["plain","line\n","\u4f60\u597d"]})", "values");
    check(static_cast<bool>(strings));
    check(strings.value().size() == 3);
    check(strings.value()[0] == "plain");
    check(strings.value()[1] == "line\n");
    check(strings.value()[2] == "\xE4\xBD\xA0\xE5\xA5\xBD");
    auto owned_strings = [] {
        const std::string source = R"({"values":["escaped\tvalue","\u4f60\u597d"]})";
        return read_manifest_string_array(source, "values");
    }();
    check(static_cast<bool>(owned_strings));
    check(owned_strings.value() == std::vector<std::string>{"escaped\tvalue", "\xE4\xBD\xA0\xE5\xA5\xBD"});
    for (const char* array : {"[1]", R"(["x"junk])", R"(["\q"])", "[,]"})
    {
        auto invalid = read_manifest_string_array(std::string(R"({"values":)") + array + "}", "values");
        check(!invalid);
        check(invalid.error().code == ErrorCode::InvalidModel);
    }

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
    GptOssModelAdapter adapter;
    ModelManifest manifest;
    manifest.model_type = "gpt_oss";
    check(adapter.can_load(manifest));

    ModelPackage gpt_package;
    gpt_package.manifest.model_type = "gpt_oss";
    gpt_package.manifest.raw_json = R"({
        "vocab_size": 128,
        "hidden_size": 16,
        "intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_local_experts": 4,
        "experts_per_token": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "sliding_window": 128,
        "initial_context_length": 128,
        "max_position_embeddings": 128,
        "rms_norm_eps": 0.00001,
        "rope_theta": 150000.0,
        "rope_scaling": {
            "factor": 24.0,
            "beta_slow": 0.5,
            "beta_fast": 16.0
        },
        "factor": 3.0,
        "attention_bias": false
    })";
    auto gpt_descriptor = adapter.parse_model(gpt_package);
    check(static_cast<bool>(gpt_descriptor));
    check(gpt_descriptor.value().layers.size() == 1);
    check_near(gpt_descriptor.value().layers.front().attention.rope_scaling_factor,
               24.0f,
               1e-6f);

    ModelPackage invalid_gpt_rope = gpt_package;
    invalid_gpt_rope.manifest.raw_json = std::regex_replace(invalid_gpt_rope.manifest.raw_json,
                                                            std::regex(R"("rope_scaling"\s*:\s*\{[^{}]*\})"),
                                                            R"("rope_scaling":[1])");
    auto invalid_gpt_rope_result = adapter.parse_model(invalid_gpt_rope);
    check(!invalid_gpt_rope_result);
    check(invalid_gpt_rope_result.error().message
          == "invalid GPT-OSS object field: rope_scaling");

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
        ScopedTestDirectory directory("ncnn_moe_model_directory_test_");
        TestRuntime runtime;
        const std::string expected = "no config.json or GGUF model files found directly in model directory: "
                                     + std::filesystem::absolute(directory.path()).string()
                                     + "; subdirectories are not searched; select the directory containing model files or a GGUF file";
        const auto reject_directory = [&]() {
            auto loaded = runtime.load_model(directory.path());
            check(!loaded && loaded.error().code == ErrorCode::InvalidArgument);
            check(loaded.error().message == expected);
        };
        reject_directory();

        const auto child = directory.path() / "package";
        std::filesystem::create_directory(child);
        {
            // Neither a child manifest nor a child GGUF file makes the parent
            // a model directory. Its contents must never be opened here.
            std::ofstream manifest(child / "config.json", std::ios::binary);
            manifest << R"({"model_type":"test_moe"})";
            std::ofstream gguf(child / "model.gguf", std::ios::binary);
            gguf << "not opened";
        }
        reject_directory();
        std::filesystem::create_directory(directory.path() / "config.json");
        reject_directory();

        const auto explicit_manifest = directory.path() / "missing.json";
        auto missing_manifest = runtime.load_model(explicit_manifest);
        check(!missing_manifest && missing_manifest.error().code == ErrorCode::IoError);
        check(missing_manifest.error().message == "cannot open manifest: " + std::filesystem::absolute(explicit_manifest).string());
    }
    {
        TemporaryModelPackage package;
        TestRuntime runtime;
        auto empty_path = runtime.load_model(std::filesystem::path{});
        check(!empty_path);
        check(empty_path.error().code == ErrorCode::InvalidArgument);
        Option isolation_options;
        isolation_options.hybrid_mode = HybridMode::CpuOnly;
        auto absolute_model = runtime.load_model(package.path(), isolation_options);
        check(static_cast<bool>(absolute_model));
        check(absolute_model.value()->stop_tokens().empty());
        auto text_prompt = absolute_model.value()->encode(R"([{"role":"user","content":"hello"}])");
        check(!text_prompt && text_prompt.error().code == ErrorCode::UnsupportedModel);
        std::string pending = "unchanged";
        auto text_token = absolute_model.value()->decode(0, pending);
        check(!text_token && text_token.error().code == ErrorCode::UnsupportedModel);
        check(pending == "unchanged");

        TemporaryModelPackage invalid_package;
        invalid_package.write_manifest("unknown_family");
        auto failed_load = runtime.load_model(invalid_package.path());
        check(!failed_load);
        check(failed_load.error().code == ErrorCode::UnsupportedModel);
        TemporaryModelPackage truncated_package;
        truncated_package.truncate_weights();
        auto truncated_load = runtime.load_model(truncated_package.path());
        check(!truncated_load);
        check(truncated_load.error().code == ErrorCode::InvalidModel);
        auto existing_session = runtime.create_session(absolute_model.value());
        check(static_cast<bool>(existing_session));
        const std::array<int32_t, 1> isolation_prompt = {0};
        auto existing_prefill = existing_session.value()->prefill(isolation_prompt);
        check(static_cast<bool>(existing_prefill));
        auto subsequent_model = runtime.load_model(package.path(), isolation_options);
        check(static_cast<bool>(subsequent_model));
        auto subsequent_session = runtime.create_session(subsequent_model.value());
        check(static_cast<bool>(subsequent_session));
        auto subsequent_prefill = subsequent_session.value()->prefill(isolation_prompt);
        check(static_cast<bool>(subsequent_prefill));
        check(existing_prefill.value().logits.size() == subsequent_prefill.value().logits.size());
        for (size_t index = 0; index < existing_prefill.value().logits.size(); ++index)
            check_near(existing_prefill.value().logits[index], subsequent_prefill.value().logits[index], 1e-6f);

        std::error_code cwd_error;
        const std::filesystem::path original_cwd = std::filesystem::current_path(cwd_error);
        check(!cwd_error);
        try
        {
            std::filesystem::current_path(package.path().parent_path(), cwd_error);
            check(!cwd_error);
            auto relative_directory = runtime.load_model(package.path().filename());
            check(static_cast<bool>(relative_directory));
            auto relative_config = runtime.load_model(package.path().filename() / "config.json");
            check(static_cast<bool>(relative_config));
        }
        catch (...)
        {
            std::error_code restore_error;
            std::filesystem::current_path(original_cwd, restore_error);
            throw;
        }
        std::filesystem::current_path(original_cwd, cwd_error);
        check(!cwd_error);
    }

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
        TestRuntime runtime;

        Option invalid_hybrid;
        invalid_hybrid.hybrid_mode = static_cast<HybridMode>(255);
        auto hybrid_model = runtime.load_model(package.path(), invalid_hybrid);
        check(static_cast<bool>(!hybrid_model));
        check(static_cast<bool>(hybrid_model.error().code == ErrorCode::InvalidArgument));

        Option invalid_packed_weights;
        invalid_packed_weights.cpu_packed_weight_mode = static_cast<CpuPackedWeightMode>(255);
        auto packed_model = runtime.load_model(package.path(), invalid_packed_weights);
        check(static_cast<bool>(!packed_model));
        check(static_cast<bool>(packed_model.error().code == ErrorCode::InvalidArgument));

        Option invalid_expert_io;
        invalid_expert_io.expert_io_mode = static_cast<ExpertIoMode>(255);
        auto expert_io_model = runtime.load_model(package.path(), invalid_expert_io);
        check(static_cast<bool>(!expert_io_model));
        check(static_cast<bool>(expert_io_model.error().code == ErrorCode::InvalidArgument));
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
        options.hybrid_mode = HybridMode::CpuOnly;
        options.expert_memory_mode = ExpertMemoryMode::Eager;
        options.expert_gpu_cache_size = 64 * 1024 * 1024;
        auto model = runtime.load_model(package.path(), options);
        check(static_cast<bool>(!model));
        check(static_cast<bool>(model.error().code == ErrorCode::UnsupportedModel));
        check(model.error().message == "the Expert GPU cache requires Vulkan mixed execution");
    }

    {
        TemporaryModelPackage package;
        TestRuntime runtime;
        Option options;
        options.hybrid_mode = HybridMode::CpuOnly;
        options.expert_memory_mode = ExpertMemoryMode::Eager;
        options.expert_gpu_victim_cache_size = 64 * 1024 * 1024;
        auto model = runtime.load_model(package.path(), options);
        check(static_cast<bool>(!model));
        check(static_cast<bool>(model.error().code == ErrorCode::UnsupportedModel));
        check(model.error().message == "the Expert GPU cache requires Vulkan mixed execution");
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
    const bool automatic_uses_vulkan = has_usable_vulkan_device(runtime.info(), RuntimeVulkanCpu);
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
    check(static_cast<bool>(!has_flag(automatic_effective.optimization_flags,
                                      OptimizationCpuPackedWeights)));
    check(static_cast<bool>(automatic_effective.num_concurrent_sessions == automatic_options.num_concurrent_sessions));
    check(static_cast<bool>(automatic_effective.expert_io_mode == automatic_options.expert_io_mode));
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

        std::vector<uint32_t> usable_device_indices;
        for (const GpuInfo& gpu : runtime.info().gpu_infos)
        {
            if (gpu.type != VulkanDeviceType::Cpu && gpu.compute_queue_count > 0)
                usable_device_indices.push_back(gpu.device_index);
        }
        if (usable_device_indices.size() >= 2)
        {
            Option multi_device_options;
            multi_device_options.hybrid_mode = HybridMode::HybridExperts;
            multi_device_options.vulkan_device_indices = {usable_device_indices[0], usable_device_indices[1]};
            auto multi_device_model = runtime.load_model(package.path(), multi_device_options);
            check(static_cast<bool>(multi_device_model));
            check(multi_device_model.value()->vulkan_device_index() == usable_device_indices[0]);
            const std::vector<uint32_t>& selected_devices = multi_device_model.value()->vulkan_device_indices();
            check(!selected_devices.empty());
            check(std::find(selected_devices.begin(), selected_devices.end(), usable_device_indices[0]) != selected_devices.end());
            for (uint32_t device_index : selected_devices)
                check(device_index == usable_device_indices[0] || device_index == usable_device_indices[1]);
            const CompiledModel& compiled = model_compiled(*multi_device_model.value());
            check(std::find(selected_devices.begin(), selected_devices.end(),
                            compiled.net.layer_plans.front().vulkan_device_index)
                  != selected_devices.end());
        }

        const SessionStatistics& statistics = automatic_session.value()->statistics();
        check(static_cast<bool>(statistics.vulkan_linear_dispatches == 1));
        check(static_cast<bool>(statistics.vulkan_compute_submissions == 1));
        check(static_cast<bool>(statistics.vulkan_batch_uploads == 1));
        check(static_cast<bool>(statistics.vulkan_batch_downloads == 1));
    }
    else
        check(static_cast<bool>(automatic_session.value()->statistics().vulkan_linear_dispatches == 0));

    Option cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto cpu_model = runtime.load_model(package.path(), cpu_options);
    check(static_cast<bool>(cpu_model));
    check(static_cast<bool>(cpu_model.value()->vulkan_device_index() == automatic_vulkan_device_index));
    const CompiledModel& cpu_compiled = model_compiled(*cpu_model.value());
    check(!cpu_compiled.vulkan_runtime);
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
    if (has_usable_vulkan_device(runtime.info(), RuntimeVulkanCpu))
    {
        check(static_cast<bool>(hybrid_model));
        check(static_cast<bool>(hybrid_model.value()->hybrid_mode() == HybridMode::HybridExperts));

        AttentionPackage attention_package;
        Option attention_options = hybrid_options;
        attention_options.optimization_flags |= OptimizationVulkanDecodeSdpa;
        auto attention_model = runtime.load_model(attention_package.path(),
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
            // A four-column projection permits pack4 output while a long
            // host prefill retains an unpacked residual.
            AttentionPackage replay_package(false, 2, true, true, true, 1024);
            auto replay_model = runtime.load_model(replay_package.path(), attention_options);
            auto cpu_replay_model = runtime.load_model(replay_package.path(), cpu_options);
            check(static_cast<bool>(replay_model));
            check(static_cast<bool>(cpu_replay_model));
            SessionOptions replay_options;
            replay_options.prefill_chunk_size = 512;
            auto replay = runtime.create_session(replay_model.value(), replay_options);
            auto cpu_replay = runtime.create_session(cpu_replay_model.value(), replay_options);
            check(static_cast<bool>(replay));
            check(static_cast<bool>(cpu_replay));
            std::vector<int32_t> prompt(512);
            for (size_t index = 0; index < prompt.size(); ++index)
                prompt[index] = static_cast<int32_t>(index % 2);
            for (uint32_t pass = 0; pass < 2; ++pass)
            {
                check(static_cast<bool>(replay.value()->reset()));
                check(static_cast<bool>(cpu_replay.value()->reset()));
                auto prefill = replay.value()->prefill(prompt);
                auto cpu_prefill = cpu_replay.value()->prefill(prompt);
                if (!prefill)
                    throw std::runtime_error("512-token Vulkan Attention replay prefill failed (pass "
                                             + std::to_string(pass) + "): " + prefill.error().message);
                if (!cpu_prefill)
                    throw std::runtime_error("512-token CPU Attention replay prefill failed (pass "
                                             + std::to_string(pass) + "): " + cpu_prefill.error().message);
                check(prefill.value().logits.size() == cpu_prefill.value().logits.size());
                for (size_t index = 0; index < cpu_prefill.value().logits.size(); ++index)
                    check_near(prefill.value().logits[index], cpu_prefill.value().logits[index], 1e-4f);
                check(replay.value()->statistics().vulkan_attention_blocks == 1);
                check(replay.value()->statistics().vulkan_attention_cpu_fallbacks == 0);

                auto decode = replay.value()->decode(1);
                auto cpu_decode = cpu_replay.value()->decode(1);
                if (!decode)
                    throw std::runtime_error("512-token Vulkan Attention replay decode failed (pass "
                                             + std::to_string(pass) + "): " + decode.error().message);
                if (!cpu_decode)
                    throw std::runtime_error("512-token CPU Attention replay decode failed (pass "
                                             + std::to_string(pass) + "): " + cpu_decode.error().message);
                check(decode.value().logits.size() == cpu_decode.value().logits.size());
                for (size_t index = 0; index < cpu_decode.value().logits.size(); ++index)
                    check_near(decode.value().logits[index], cpu_decode.value().logits[index], 1e-4f);
                check(replay.value()->statistics().vulkan_attention_blocks == 2);
                check(replay.value()->statistics().vulkan_attention_cpu_fallbacks == 0);
            }
        }

        {
            SessionState gpu_state;
            SessionState cpu_state;
            SessionStatistics gpu_statistics;
            SessionStatistics cpu_statistics;
            const CompiledModel& gpu_compiled = model_compiled(*attention_model.value());
            const CompiledModel& cpu_attention_compiled = model_compiled(*cpu_attention_model.value());
            check(static_cast<bool>(forward_model(gpu_compiled, attention_prompt, gpu_statistics, gpu_state, 0)));
            check(static_cast<bool>(forward_model(cpu_attention_compiled, attention_prompt, cpu_statistics, cpu_state, 0)));
            check(gpu_statistics.vulkan_attention_blocks == 1);
            check(!gpu_state.layers.empty());
            const auto cache = gpu_state.layers.front().vulkan_attention_cache;
            check(static_cast<bool>(cache));
            const uint64_t capacity = gpu_state.layers.front().capacity_tokens;
            check(capacity > attention_prompt.size());

            const std::array<int32_t, 1> next_token = {0};
            auto gpu_decode_status = forward_model(gpu_compiled, next_token, gpu_statistics, gpu_state, attention_prompt.size());
            auto cpu_decode_status = forward_model(cpu_attention_compiled, next_token, cpu_statistics, cpu_state, attention_prompt.size());
            check(static_cast<bool>(gpu_decode_status));
            check(static_cast<bool>(cpu_decode_status));
            check(gpu_statistics.vulkan_attention_blocks == 2);
            check(gpu_state.layers.front().vulkan_attention_cache == cache);
            check(gpu_state.layers.front().capacity_tokens == capacity);
            check(!gpu_state.layers.front().vulkan_attention_state_unknown);
            check(gpu_state.logits.rows() == 1);
            check(cpu_state.logits.rows() == 1);
            check(gpu_state.logits.columns() == cpu_state.logits.columns());
            for (size_t index = 0; index < cpu_state.logits.columns(); ++index)
                check_near(gpu_state.logits.row(0)[index], cpu_state.logits.row(0)[index], 1e-4f);
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
            check(static_cast<bool>(control_statistics.vulkan_attention_device_rope_fusions == 0));
            check(static_cast<bool>(control_attention_prefill.value().logits.size()
                                    == attention_prefill.value().logits.size()));
            for (size_t index = 0;
                 index < attention_prefill.value().logits.size();
                 ++index)
            {
                check_near(control_attention_prefill.value().logits[index],
                           attention_prefill.value().logits[index],
                           1e-4f);
            }
        }

        auto check_attention_cache_isolation =
            [&](const std::filesystem::path& package_path) {
                Option disabled_attention_options = hybrid_options;
                disabled_attention_options.optimization_flags &= ~OptimizationVulkanAttention;
                auto promotion_model = runtime.load_model(package_path,
                                                          disabled_attention_options);
                auto promotion_cpu_model = runtime.load_model(package_path, cpu_options);
                check(static_cast<bool>(promotion_model));
                check(static_cast<bool>(promotion_cpu_model));
                auto promotion_session = runtime.create_session(promotion_model.value());
                auto promotion_cpu_session = runtime.create_session(promotion_cpu_model.value());
                check(static_cast<bool>(promotion_session));
                check(static_cast<bool>(promotion_cpu_session));
                check(static_cast<bool>(promotion_session.value()->prefill(attention_prompt)));
                check(static_cast<bool>(promotion_cpu_session.value()->prefill(attention_prompt)));
                check(static_cast<bool>(promotion_session.value()->statistics().vulkan_attention_blocks
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
                    check_near(promoted_decode.value().logits[index],
                               promotion_cpu_decode.value().logits[index],
                               1e-4f);
                }
                const SessionStatistics& promoted_statistics = promotion_session.value()->statistics();
                check(static_cast<bool>(promoted_statistics.vulkan_attention_blocks == 0));

                auto continued_decode = promotion_session.value()->decode(0);
                auto continued_cpu_decode = promotion_cpu_session.value()->decode(0);
                check(static_cast<bool>(continued_decode));
                check(static_cast<bool>(continued_cpu_decode));
                for (size_t index = 0;
                     index
                     < continued_cpu_decode.value().logits.size();
                     ++index)
                {
                    check_near(continued_decode.value().logits[index],
                               continued_cpu_decode.value().logits[index],
                               1e-4f);
                }
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
        check(static_cast<bool>(wrapped_statistics.vulkan_attention_decode_sdpa_fusions <= wrapped_statistics.vulkan_attention_blocks - 1));
        check(static_cast<bool>(wrapped_statistics.vulkan_kv_ring_resizes == 1));
        check(static_cast<bool>(wrapped_statistics.vulkan_kv_ring_wrapped_views > 0));

        for (const auto& [window, sinks] : std::array<std::pair<uint32_t, bool>, 3>{{{2, true}, {2, false}, {0, true}}})
        {
            AttentionPackage gqa_package(false, window, true, sinks, true);
            auto gqa_model = runtime.load_model(gqa_package.path(), attention_options);
            auto gqa_cpu_model = runtime.load_model(gqa_package.path(), cpu_options);
            check(static_cast<bool>(gqa_model));
            check(static_cast<bool>(gqa_cpu_model));
            const AttentionDescriptor& descriptor = model_compiled(*gqa_model.value()).descriptor.layers.front().attention;
            check(descriptor.kind == AttentionKind::Standard);
            check(descriptor.head_count == 2 && descriptor.kv_head_count == 1);
            check((descriptor.sliding_window != 0) == (window != 0));
            check(has_flag(descriptor.flags, AttentionDescriptorSinks) == sinks);

            auto gqa_session = runtime.create_session(gqa_model.value());
            auto gqa_cpu_session = runtime.create_session(gqa_cpu_model.value());
            check(static_cast<bool>(gqa_session));
            check(static_cast<bool>(gqa_cpu_session));
            auto gqa_prefill = gqa_session.value()->prefill(attention_prompt);
            auto gqa_cpu_prefill = gqa_cpu_session.value()->prefill(attention_prompt);
            if (!gqa_prefill)
                throw std::runtime_error("GQA prefill failed (window=" + std::to_string(window)
                                         + ", sinks=" + std::to_string(sinks) + "): "
                                         + gqa_prefill.error().message);
            check(static_cast<bool>(gqa_cpu_prefill));
            check(gqa_prefill.value().logits.size() == gqa_cpu_prefill.value().logits.size());
            for (size_t index = 0; index < gqa_cpu_prefill.value().logits.size(); ++index)
                check_near(gqa_prefill.value().logits[index], gqa_cpu_prefill.value().logits[index], 1e-4f);
            check(gqa_session.value()->statistics().vulkan_attention_blocks == 1);
            check(gqa_session.value()->statistics().vulkan_attention_device_rope_fusions > 0);
            check(gqa_session.value()->statistics().vulkan_attention_cpu_fallbacks == 0);

            for (uint32_t iteration = 0; iteration < 40; ++iteration)
            {
                const int32_t token = static_cast<int32_t>(iteration % 2);
                auto gqa_decode = gqa_session.value()->decode(token);
                auto gqa_cpu_decode = gqa_cpu_session.value()->decode(token);
                if (!gqa_decode)
                    throw std::runtime_error("GQA decode failed (window=" + std::to_string(window)
                                             + ", sinks=" + std::to_string(sinks)
                                             + ", step=" + std::to_string(iteration) + "): "
                                             + gqa_decode.error().message);
                check(static_cast<bool>(gqa_cpu_decode));
                check(gqa_decode.value().logits.size() == gqa_cpu_decode.value().logits.size());
                for (size_t index = 0; index < gqa_cpu_decode.value().logits.size(); ++index)
                    check_near(gqa_decode.value().logits[index], gqa_cpu_decode.value().logits[index], 1e-4f);
            }
            const SessionStatistics& gqa_statistics = gqa_session.value()->statistics();
            check(gqa_statistics.vulkan_attention_blocks > 1);
            check(gqa_statistics.vulkan_attention_device_rope_fusions > 0);
            check(gqa_statistics.vulkan_attention_qkv_ring_fusions > 0);
            check(gqa_statistics.vulkan_kv_ring_appends > 0);
            if (window != 0)
                check(gqa_statistics.vulkan_kv_ring_wrapped_views > 0);
            else
                check(gqa_statistics.vulkan_kv_ring_resizes > 1);
            check(gqa_statistics.vulkan_attention_cpu_fallbacks == 0);
            const SessionStatistics& gqa_cpu_statistics = gqa_cpu_session.value()->statistics();
            check(gqa_cpu_statistics.vulkan_attention_blocks == 0);
            check(gqa_cpu_statistics.vulkan_attention_cpu_fallbacks == 0);

            if (window == 0)
            {
                const CompiledModel& gqa_compiled = model_compiled(*gqa_model.value());
                const CompiledModel& gqa_cpu_compiled = model_compiled(*gqa_cpu_model.value());
                const AttentionBlockPlan& gqa_plan = gqa_compiled.net.layer_plans.front().attention;
                const AttentionBlockPlan& gqa_cpu_plan = gqa_cpu_compiled.net.layer_plans.front().attention;
                const uint32_t hidden_size = gqa_compiled.descriptor.hidden_size;
                ActivationBuffer prefix(2, hidden_size);
                ActivationBuffer transaction_input(3, hidden_size);
                for (size_t row = 0; row < prefix.rows(); ++row)
                    for (uint32_t column = 0; column < hidden_size; ++column)
                        prefix.row(row)[column] = static_cast<float>(static_cast<int>((row * 7 + column * 3) % 11) - 5) / 16.0f;
                for (size_t row = 0; row < transaction_input.rows(); ++row)
                    for (uint32_t column = 0; column < hidden_size; ++column)
                        transaction_input.row(row)[column] = static_cast<float>(static_cast<int>((row * 5 + column * 7 + 2) % 13) - 6) / 32.0f;
                const auto compare_attention_outputs = [](const ActivationBuffer& actual,
                                                          const ActivationBuffer& expected) {
                    check(actual.rows() == expected.rows());
                    check(actual.columns() == expected.columns());
                    for (size_t row = 0; row < expected.rows(); ++row)
                        for (uint32_t column = 0; column < expected.columns(); ++column)
                            check_near(actual.row(row)[column], expected.row(row)[column], 1e-4f);
                };

                for (size_t committed_rows : {size_t{1}, size_t{0}})
                {
                    std::array<LayerCache, 1> hybrid_cache;
                    LayerCache reference_cache;
                    AttentionScratch hybrid_scratch;
                    AttentionScratch reference_scratch;
                    ActivationBuffer hybrid_output;
                    ActivationBuffer reference_output;
                    const VulkanStatistics before_prefix = get_vulkan_statistics(gqa_compiled.vulkan_runtime);
                    check(static_cast<bool>(forward_attention(gqa_compiled.weights, gqa_compiled.operators, gqa_plan,
                                                              ExecutionBackend::Vulkan, 0, hybrid_cache.front(),
                                                              hybrid_scratch, prefix, hybrid_output,
                                                              gqa_compiled.opt.optimization_flags)));
                    check(static_cast<bool>(forward_attention(gqa_cpu_compiled.weights, gqa_cpu_compiled.operators,
                                                              gqa_cpu_plan, ExecutionBackend::Cpu, 0,
                                                              reference_cache, reference_scratch, prefix,
                                                              reference_output, gqa_cpu_compiled.opt.optimization_flags)));
                    compare_attention_outputs(hybrid_output, reference_output);
                    check(get_vulkan_statistics(gqa_compiled.vulkan_runtime).attention_blocks
                          > before_prefix.attention_blocks);

                    check(static_cast<bool>(begin_state_cache_transaction(hybrid_cache,
                                                                          transaction_input.rows())));
                    const VulkanStatistics before_transaction = get_vulkan_statistics(gqa_compiled.vulkan_runtime);
                    check(static_cast<bool>(forward_attention(gqa_compiled.weights, gqa_compiled.operators, gqa_plan,
                                                              ExecutionBackend::Vulkan, prefix.rows(),
                                                              hybrid_cache.front(), hybrid_scratch,
                                                              transaction_input, hybrid_output,
                                                              gqa_compiled.opt.optimization_flags)));
                    check(get_vulkan_statistics(gqa_compiled.vulkan_runtime).attention_blocks
                          > before_transaction.attention_blocks);
                    check(static_cast<bool>(finish_state_cache_transaction(hybrid_cache, committed_rows)));
                    if (committed_rows != 0)
                    {
                        ActivationBuffer committed_input(committed_rows, hidden_size);
                        std::copy_n(transaction_input.row(0), committed_input.values().size(),
                                    committed_input.row(0));
                        check(static_cast<bool>(forward_attention(gqa_cpu_compiled.weights,
                                                                  gqa_cpu_compiled.operators, gqa_cpu_plan,
                                                                  ExecutionBackend::Cpu, prefix.rows(),
                                                                  reference_cache, reference_scratch,
                                                                  committed_input, reference_output,
                                                                  gqa_cpu_compiled.opt.optimization_flags)));
                    }
                    check(hybrid_cache.front().token_count == prefix.rows() + committed_rows);
                    check(hybrid_cache.front().token_count == reference_cache.token_count);
                    check(hybrid_cache.front().start_position == reference_cache.start_position);
                    check(hybrid_cache.front().first_slot == reference_cache.first_slot);

                    ActivationBuffer continuation(1, hidden_size);
                    std::copy_n(transaction_input.row(committed_rows), hidden_size, continuation.row(0));
                    const uint64_t continuation_position = prefix.rows() + committed_rows;
                    check(static_cast<bool>(forward_attention(gqa_compiled.weights, gqa_compiled.operators, gqa_plan,
                                                              ExecutionBackend::Vulkan, continuation_position,
                                                              hybrid_cache.front(), hybrid_scratch,
                                                              continuation, hybrid_output,
                                                              gqa_compiled.opt.optimization_flags)));
                    check(static_cast<bool>(forward_attention(gqa_cpu_compiled.weights, gqa_cpu_compiled.operators,
                                                              gqa_cpu_plan, ExecutionBackend::Cpu,
                                                              continuation_position, reference_cache,
                                                              reference_scratch, continuation, reference_output,
                                                              gqa_cpu_compiled.opt.optimization_flags)));
                    compare_attention_outputs(hybrid_output, reference_output);
                    check(hybrid_cache.front().token_count == reference_cache.token_count);
                }

                CompiledModel isolated_compiled = gqa_compiled;
                AttentionBlockPlan& isolated_plan = isolated_compiled.net.layer_plans.front().attention;
                const CompiledOperatorHandle attention_handle = isolated_plan.vulkan_attention_operator;
                check(attention_handle != invalid_compiled_operator_handle);
                const auto saved_attention_operator = isolated_compiled.operators.at(attention_handle).attention;
                check(saved_attention_operator != nullptr);
                LayerCache seeded_cache;
                AttentionScratch isolated_scratch;
                ActivationBuffer isolated_output;
                LayerCache isolated_cpu_cache;
                AttentionScratch isolated_cpu_scratch;
                ActivationBuffer isolated_cpu_output;
                check(static_cast<bool>(forward_attention(isolated_compiled.weights, isolated_compiled.operators,
                                                          isolated_plan, ExecutionBackend::Vulkan, 0,
                                                          seeded_cache, isolated_scratch, prefix,
                                                          isolated_output, isolated_compiled.opt.optimization_flags)));
                check(static_cast<bool>(forward_attention(gqa_cpu_compiled.weights, gqa_cpu_compiled.operators,
                                                          gqa_cpu_plan, ExecutionBackend::Cpu, 0,
                                                          isolated_cpu_cache, isolated_cpu_scratch, prefix,
                                                          isolated_cpu_output, gqa_cpu_compiled.opt.optimization_flags)));
                compare_attention_outputs(isolated_output, isolated_cpu_output);

                isolated_compiled.operators.at_mutable(attention_handle).attention.reset();
                LayerCache empty_cache;
                AttentionScratch failed_scratch;
                ActivationBuffer failed_output;
                const VulkanStatistics before_missing_prefill = get_vulkan_statistics(isolated_compiled.vulkan_runtime);
                auto missing_prefill = forward_attention(isolated_compiled.weights, isolated_compiled.operators,
                                                         isolated_plan, ExecutionBackend::Vulkan, 0,
                                                         empty_cache, failed_scratch, prefix, failed_output,
                                                         isolated_compiled.opt.optimization_flags);
                check(!missing_prefill);
                check(missing_prefill.error().code == ErrorCode::InternalError);
                check(missing_prefill.error().message
                      == "Vulkan Standard Attention was selected without a complete attention operator");
                check(empty_cache.token_count == 0);
                const VulkanStatistics after_missing_prefill = get_vulkan_statistics(isolated_compiled.vulkan_runtime);
                check(after_missing_prefill.attention_blocks == before_missing_prefill.attention_blocks);
                check(after_missing_prefill.attention_cpu_fallbacks == before_missing_prefill.attention_cpu_fallbacks);

                ActivationBuffer decode_input(1, hidden_size);
                std::copy_n(transaction_input.row(0), hidden_size, decode_input.row(0));
                const VulkanStatistics before_missing_decode = get_vulkan_statistics(isolated_compiled.vulkan_runtime);
                auto missing_decode = forward_attention(isolated_compiled.weights, isolated_compiled.operators,
                                                        isolated_plan, ExecutionBackend::Vulkan, prefix.rows(),
                                                        seeded_cache, failed_scratch, decode_input, failed_output,
                                                        isolated_compiled.opt.optimization_flags);
                check(!missing_decode);
                check(missing_decode.error().code == ErrorCode::InternalError);
                check(missing_decode.error().message
                      == "Vulkan Standard Attention was selected without a complete attention operator");
                check(seeded_cache.token_count == prefix.rows());
                const VulkanStatistics after_missing_decode = get_vulkan_statistics(isolated_compiled.vulkan_runtime);
                check(after_missing_decode.attention_blocks == before_missing_decode.attention_blocks);
                check(after_missing_decode.attention_cpu_fallbacks == before_missing_decode.attention_cpu_fallbacks);

                isolated_compiled.operators.at_mutable(attention_handle).attention = saved_attention_operator;
                ActivationBuffer empty_input;
                const VulkanStatistics before_failed_execution = get_vulkan_statistics(isolated_compiled.vulkan_runtime);
                auto failed_execution = forward_attention(isolated_compiled.weights, isolated_compiled.operators,
                                                          isolated_plan, ExecutionBackend::Vulkan, 0,
                                                          empty_cache, failed_scratch, empty_input, failed_output,
                                                          isolated_compiled.opt.optimization_flags);
                check(!failed_execution);
                check(failed_execution.error().code == ErrorCode::InternalError);
                check(failed_execution.error().message == "Vulkan Attention execution failed");
                check(get_vulkan_statistics(isolated_compiled.vulkan_runtime).attention_cpu_fallbacks
                      == before_failed_execution.attention_cpu_fallbacks);
                // Fresh LayerCache state mirrors resetting a Session after a failed execution.
                LayerCache recovered_cache;
                LayerCache recovered_cpu_cache;
                AttentionScratch recovered_scratch;
                AttentionScratch recovered_cpu_scratch;
                ActivationBuffer recovered_output;
                ActivationBuffer recovered_cpu_output;
                check(static_cast<bool>(forward_attention(isolated_compiled.weights, isolated_compiled.operators,
                                                          isolated_plan, ExecutionBackend::Vulkan, 0,
                                                          recovered_cache, recovered_scratch, prefix,
                                                          recovered_output, isolated_compiled.opt.optimization_flags)));
                check(static_cast<bool>(forward_attention(gqa_cpu_compiled.weights, gqa_cpu_compiled.operators,
                                                          gqa_cpu_plan, ExecutionBackend::Cpu, 0,
                                                          recovered_cpu_cache, recovered_cpu_scratch, prefix,
                                                          recovered_cpu_output, gqa_cpu_compiled.opt.optimization_flags)));
                compare_attention_outputs(recovered_output, recovered_cpu_output);
                check(static_cast<bool>(forward_attention(isolated_compiled.weights, isolated_compiled.operators,
                                                          isolated_plan, ExecutionBackend::Vulkan, prefix.rows(),
                                                          recovered_cache, recovered_scratch, decode_input,
                                                          recovered_output, isolated_compiled.opt.optimization_flags)));
                check(static_cast<bool>(forward_attention(gqa_cpu_compiled.weights, gqa_cpu_compiled.operators,
                                                          gqa_cpu_plan, ExecutionBackend::Cpu, prefix.rows(),
                                                          recovered_cpu_cache, recovered_cpu_scratch, decode_input,
                                                          recovered_cpu_output, gqa_cpu_compiled.opt.optimization_flags)));
                compare_attention_outputs(recovered_output, recovered_cpu_output);
            }
        }

        Option ncnn_sdpa_ring_options = hybrid_options;
        ncnn_sdpa_ring_options.optimization_flags |= OptimizationVulkanQkvRing;
        ncnn_sdpa_ring_options.optimization_flags &= ~OptimizationVulkanDecodeSdpa;
        auto ncnn_sdpa_ring_model = runtime.load_model(attention_package.path(),
                                                       ncnn_sdpa_ring_options);
        check(static_cast<bool>(ncnn_sdpa_ring_model));
        auto ncnn_sdpa_ring_session = runtime.create_session(ncnn_sdpa_ring_model.value());
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
        auto full_attention_model = runtime.load_model(full_attention_package.path(),
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
        // The first eligible decode uses the fixed Decode SDPA policy; there is no warm-up probe.
        check(static_cast<bool>(chunked_attention_session.value()->statistics().vulkan_attention_decode_sdpa_fusions == 1));

        Option unfused_ring_options = attention_options;
        unfused_ring_options.optimization_flags &= ~OptimizationVulkanQkvRing;
        auto unfused_ring_model = runtime.load_model(full_attention_package.path(),
                                                     unfused_ring_options);
        check(static_cast<bool>(unfused_ring_model));
        auto unfused_ring_session = runtime.create_session(unfused_ring_model.value());
        check(static_cast<bool>(unfused_ring_session));
        auto unfused_ring_prefill = unfused_ring_session.value()->prefill(attention_prompt);
        check(static_cast<bool>(unfused_ring_prefill));
        check(static_cast<bool>(unfused_ring_prefill.value().logits.size() == full_cpu_prefill.value().logits.size()));
        for (size_t index = 0; index < full_cpu_prefill.value().logits.size(); ++index)
        {
            check_near(unfused_ring_prefill.value().logits[index], full_cpu_prefill.value().logits[index], 1e-4f);
        }
        auto unfused_ring_decode = unfused_ring_session.value()->decode(1);
        check(static_cast<bool>(unfused_ring_decode));
        check(static_cast<bool>(unfused_ring_decode.value().logits.size() == full_cpu_decode.value().logits.size()));
        for (size_t index = 0; index < full_cpu_decode.value().logits.size(); ++index)
        {
            check_near(unfused_ring_decode.value().logits[index], full_cpu_decode.value().logits[index], 1e-4f);
        }
        check(static_cast<bool>(unfused_ring_session.value()->statistics().vulkan_attention_qkv_ring_fusions == 0));
        check(static_cast<bool>(unfused_ring_session.value()->statistics().vulkan_attention_decode_sdpa_fusions == 1));

        auto prefetch_session = runtime.create_session(hybrid_model.value());
        check(static_cast<bool>(prefetch_session));
        check(static_cast<bool>(prefetch_session.value()->prefill(packed_prompt)));
    }
    else
    {
        // Vulkan CPU/software devices are not eligible for heterogeneous placement.
        check(static_cast<bool>(!hybrid_model));
        check(static_cast<bool>(hybrid_model.error().code == ErrorCode::UnsupportedModel));
    }
}

void test_flag_defaults()
{
    check(static_cast<bool>(has_flag(OptimizationDefaultFlags,
                                     OptimizationVulkanCommandGraph)));
    check(static_cast<bool>(has_flag(OptimizationDefaultFlags,
                                     OptimizationCpuLatentVectorSoftmax)));
    check(static_cast<bool>(has_flag(OptimizationDefaultFlags,
                                     OptimizationVulkanLatentInputRmsNorm)));
    check(static_cast<bool>(has_flag(OptimizationDefaultFlags,
                                     OptimizationVulkanRouteAggregation)));
    check(static_cast<bool>(has_flag(OptimizationDefaultFlags,
                                     OptimizationVulkanExpertGpuPriority)));
    check(static_cast<bool>(has_flag(OptimizationDefaultFlags,
                                     OptimizationVulkanIndexedExperts)));
    check(static_cast<bool>(has_flag(OptimizationDefaultFlags,
                                     OptimizationVulkanQnK)));
    check(static_cast<bool>(!has_flag(OptimizationDefaultFlags,
                                      OptimizationCpuPackedWeights)));
    Option runtime;
    check(static_cast<bool>(runtime.optimization_flags == OptimizationDefaultFlags));
    check(static_cast<bool>(runtime.cpu_packed_weight_mode == CpuPackedWeightMode::Disabled));
    check(static_cast<bool>(runtime.expert_io_mode == ExpertIoMode::Auto));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionRouterPrediction)));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionForwardAwareCache)));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionRankAdaptivePrefetch)));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionCrossExpertReadCoalescing)));
    check(static_cast<bool>(!has_flag(runtime.flags, OptionAsyncRouterPrediction)));

    RuntimeInfo info;
    check(static_cast<bool>(has_flag(info.flags, RuntimeCpu)));
    check(static_cast<bool>(has_flag(info.flags, RuntimeMxfp4Cpu)));
    check(static_cast<bool>(has_flag(info.flags, RuntimeCrossSession)));

    RuntimeInfo vulkan_info;
    vulkan_info.flags |= RuntimeVulkanCpu;
    check(static_cast<bool>(!has_usable_vulkan_device(vulkan_info, RuntimeVulkanCpu)));
    vulkan_info.gpu_infos.resize(1);
    vulkan_info.gpu_infos.front().type = VulkanDeviceType::Cpu;
    check(static_cast<bool>(!has_usable_vulkan_device(vulkan_info, RuntimeVulkanCpu)));
    vulkan_info.gpu_infos.front().type = VulkanDeviceType::Integrated;
    check(static_cast<bool>(has_usable_vulkan_device(vulkan_info, RuntimeVulkanCpu)));
    vulkan_info.default_gpu_index = 1;
    check(static_cast<bool>(!has_usable_vulkan_device(vulkan_info, RuntimeVulkanCpu)));
    vulkan_info.default_gpu_index = 0;
    vulkan_info.flags &= ~RuntimeVulkanCpu;
    check(static_cast<bool>(!has_usable_vulkan_device(vulkan_info, RuntimeVulkanCpu)));

    SchedulerOptions scheduler;
    check(static_cast<bool>(scheduler.num_threads == 0));
    check(static_cast<bool>(scheduler.use_staged_decode));

    ExpertDispatchOptions dispatch;
    check(dispatch.normalization == RouterNormalization::SelectedExperts);

    MoeDescriptor moe;
    check(moe.normalization == RouterNormalization::SelectedExperts);
    check(moe.shared_expert_count == 0);

    LayerDescriptor layer;
    check(layer.moe.normalization == RouterNormalization::SelectedExperts);
    check(layer.moe.shared_expert_count == 0);
    check(layer.moe.activation == ExpertActivation::Silu);
    check(layer.moe.layout == ExpertLayout::GateUpDown);
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

} // namespace moe
} // namespace ncnn
