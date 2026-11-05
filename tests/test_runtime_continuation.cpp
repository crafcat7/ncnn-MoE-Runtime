#include "test_runtime_decls.h"
#include "test_runtime_support.h"

namespace ncnn {
namespace moe {

void test_gated_delta_vulkan_budget_admission()
{
    if (get_gpu_count() == 0)
        return;

    bool bf16_supported = false;
#if NCNN_MOE_WITH_NCNN && NCNN_MOE_WITH_VULKAN
    const auto context = VulkanContext::acquire(get_default_gpu_index(), create_vulkan_runtime(), g_test_optimization_flags);
    check(static_cast<bool>(context));
    bf16_supported = context->device()->info.subgroup_size() == 32;
    if (!bf16_supported)
        std::cout << "SKIP BF16 GDN GPU admission: device subgroup size " << context->device()->info.subgroup_size()
                  << " is not 32; CPU fallback parity remains enabled\n";
#endif

    Qwen4ExpModelAdapter adapter;
    auto parsed = adapter.parse_model(qwen4_exp_package());
    check(static_cast<bool>(parsed));
    MoeModelDescriptor descriptor = parsed.value();
    descriptor.layers.resize(3);
    descriptor.hidden_size = 32;
    descriptor.intermediate_size = 32;
    descriptor.speculative_layer_count = 0;
    descriptor.speculative_kind = SpeculativeModelKind::None;
    descriptor.speculative_target_layer_ids.clear();
    for (LayerDescriptor& layer : descriptor.layers)
    {
        layer.ple = {};
        layer.moe.intermediate_size = descriptor.intermediate_size;
        layer.moe.expert_weight_dtype = DType::MxFp4;
        layer.moe.flags |= MoeDescriptorFileBackedExperts;
    }

    ScopedTestDirectory directory("ncnn_moe_gdn_budget_test_");
    const std::filesystem::path blocks_path = directory.path() / "blocks.bin";
    const std::filesystem::path scales_path = directory.path() / "scales.bin";
    {
        const std::vector<uint8_t> blocks(1024, 0);
        const std::vector<uint8_t> scales(64, 127);
        std::ofstream blocks_stream(blocks_path, std::ios::binary);
        std::ofstream scales_stream(scales_path, std::ios::binary);
        blocks_stream.write(reinterpret_cast<const char*>(blocks.data()), static_cast<std::streamsize>(blocks.size()));
        scales_stream.write(reinterpret_cast<const char*>(scales.data()), static_cast<std::streamsize>(scales.size()));
        check(static_cast<bool>(blocks_stream));
        check(static_cast<bool>(scales_stream));
    }

    const auto make_mapping = [&](const MoeModelDescriptor& model_descriptor) {
        WeightMapping result = qwen4_exp_test_mapping(model_descriptor);
        for (auto& [name, tensor] : result)
        {
            if (name.find(".experts.") != std::string::npos)
            {
                tensor.dtype = DType::MxFp4;
                tensor.bfloat16_data.clear();
                auto storage = std::make_shared<MxFp4FileStorage>();
                storage->blocks_path = blocks_path.string();
                storage->blocks_size = tensor.element_count() / 2;
                storage->scales_path = scales_path.string();
                storage->scales_size = tensor.element_count() / 32;
                tensor.mxfp4_file_storage = std::move(storage);
                continue;
            }
            if (name.find(".attention.delta.") == std::string::npos
                && name.find(".attention.output.weight") == std::string::npos)
                continue;

            for (size_t index = 0; index < tensor.bfloat16_data.size(); ++index)
            {
                float value = static_cast<float>(static_cast<int>((index * 7 + 3) % 17) - 8) / 64.0f;
                if (name.find("convolution.weight") != std::string::npos)
                    value = index % 2 == 0 ? 0.25f : 0.5f;
                else if (name.find("time_bias") != std::string::npos)
                    value = 0.125f;
                else if (name.find("decay_log") != std::string::npos)
                    value = -0.5f;
                else if (name.find("norm.weight") != std::string::npos)
                    value = 1.0f + static_cast<float>(index) / 16.0f;
                tensor.bfloat16_data[index] = float_to_bfloat16(value);
            }
        }
        return result;
    };
    WeightMapping mapping = make_mapping(descriptor);

    CompilerOption cpu_opt;
    cpu_opt.flags |= BackendFileBackedExperts;
    cpu_opt.optimization_flags = g_test_optimization_flags;
    CompiledModel cpu_model;
    check(static_cast<bool>(compile_model(descriptor, mapping, cpu_model, HybridMode::CpuOnly, cpu_opt)));

    // The continuous chain requires its complete working set. A small heap
    // must reject admission, rather than prepare only a prefix of GPU layers.
    for (bool explicit_cache : {false, true})
    {
        CompilerOption opt;
        opt.flags |= BackendVulkanDense | BackendVulkanAttention | BackendVulkanExperts | BackendFileBackedExperts;
        opt.optimization_flags = g_test_optimization_flags;
        opt.device_index = get_default_gpu_index();
        opt.vulkan_runtime = create_vulkan_runtime();
        opt.gpu_heap_budget = UINT64_C(512) * 1024;
        opt.gpu_expert_cache_reserve = explicit_cache ? UINT64_C(128) * 1024 : 0;
        CompiledModel rejected;
        auto compiled = compile_model(descriptor, mapping, rejected, HybridMode::HybridExperts, opt);
        check(!compiled);
        check(compiled.error().code == ErrorCode::InvalidArgument);
        check(compiled.error().message.find("gated residual Vulkan persistent weights and minimum working set exceed the GPU heap budget") == 0);
        check(!rejected.gated_residual_chain);
    }

    if (bf16_supported)
    {
        MoeModelDescriptor full_descriptor = descriptor;
        full_descriptor.layers.push_back(parsed.value().layers[3]);
        full_descriptor.layers.back().ple = {};
        full_descriptor.layers.back().moe.intermediate_size = full_descriptor.intermediate_size;
        full_descriptor.layers.back().moe.expert_weight_dtype = DType::MxFp4;
        full_descriptor.layers.back().moe.flags |= MoeDescriptorFileBackedExperts;
        const WeightMapping full_mapping = make_mapping(full_descriptor);
        for (bool explicit_cache : {false, true})
        {
            CompilerOption opt;
            opt.flags |= BackendVulkanDense | BackendVulkanAttention | BackendVulkanExperts | BackendFileBackedExperts;
            opt.optimization_flags = g_test_optimization_flags;
            opt.device_index = get_default_gpu_index();
            opt.vulkan_runtime = create_vulkan_runtime();
            // Zero delegates the budget query to the real Vulkan live heap;
            // this case must prepare all three GDN layers and the QSA layer.
            opt.gpu_heap_budget = 0;
            opt.gpu_expert_cache_reserve = explicit_cache ? UINT64_C(128) * 1024 : 0;
            CompiledModel model;
            auto compiled = compile_model(full_descriptor, full_mapping, model, HybridMode::HybridExperts, opt);
            if (!compiled)
                throw std::runtime_error(std::string("Qwen complete GPU chain admission failed (")
                                         + (explicit_cache ? "explicit cache" : "automatic cache") + "): " + compiled.error().message);
            check(model.gated_residual_chain);
            check(static_cast<bool>(model.gated_residual_device_head));
            const uint64_t execution_reserve = vulkan_pending_reserve(model, opt.device_index);
            check(execution_reserve > 0);
            const auto resource_plan = std::find_if(model.memory_plan.gpu_devices.begin(), model.memory_plan.gpu_devices.end(), [&](const GpuResourcePlan& plan) {
                return plan.initial.device_index == opt.device_index;
            });
            check(resource_plan != model.memory_plan.gpu_devices.end());
            check(execution_reserve == gpu_exec_reserve(*resource_plan));
            uint32_t gated_delta_layers = 0;
            uint32_t qsa_layers = 0;
            for (const CompiledLayerPlan& layer : model.net.layer_plans)
            {
                const AttentionBlockPlan& attention = layer.attention;
                check(static_cast<bool>(model.operators.at_weight(attention.output_weight).gated_residual_layer));
                check(support_vulkan_attention(model.operators, attention));
                if (attention.kind == AttentionKind::GatedDeltaNet)
                {
                    check(attention.gated_delta_vulkan_operator != invalid_compiled_operator_handle);
                    check(static_cast<bool>(model.operators.at(attention.gated_delta_vulkan_operator).gated_delta));
                    ++gated_delta_layers;
                }
                else
                {
                    check(attention.vulkan_attention_operator != invalid_compiled_operator_handle);
                    check(static_cast<bool>(model.operators.at(attention.vulkan_attention_operator).qsa_attention));
                    ++qsa_layers;
                }
            }
            check(gated_delta_layers == 3 && qsa_layers == 1);
            uint32_t gpu_expert_nodes = 0;
            for (const Layer& node : model.net.layers)
                if (node.kind == LayerKind::ExpertGroup && node.backend == ExecutionBackend::Vulkan)
                    ++gpu_expert_nodes;
            check(gpu_expert_nodes == full_descriptor.layers.size());
        }
    }
    else
        std::cout << "SKIP continuous Qwen GPU chain admission: BF16 GDN requires subgroup size 32\n";

    struct BudgetCase
    {
        const char* name;
        uint64_t heap_size;
        uint32_t sessions;
        uint64_t cache_reserve;
        bool gpu_experts;
        bool attention_backend;
        bool attention_optimization;
        bool unavailable_snapshot;
        bool expect_error;
        uint32_t expected_layers;
    };
    // The same ledger admits persistent weights, session state, and workspace.
    // Explicit caches reserve their real bytes rather than changing a GDN quota.
    constexpr uint64_t generous_heap = UINT64_C(128) * 1024 * 1024;
    const std::array<BudgetCase, 12> cases = {{
        {"automatic cache", generous_heap, 1, 0, true, true, true, false, false, 3},
        {"explicit cache", generous_heap, 1, generous_heap / 4, true, true, true, false, false, 3},
        {"cache consumes heap", generous_heap, 1, generous_heap, true, true, true, false, true, 0},
        {"no GPU Expert cache", generous_heap, 1, 0, false, true, true, false, false, 3},
        {"eight sessions share weights", generous_heap, 8, 0, true, true, true, false, false, 3},
        {"unbounded session state", generous_heap, std::numeric_limits<uint32_t>::max(), 0, true, true, true, false, true, 0},
        {"small heap", UINT64_C(8) * 1024, 1, 0, true, true, true, false, true, 0},
        {"unavailable device heap", 0, 1, 0, true, true, true, true, true, 0},
        {"automatic live heap", 0, 1, 0, true, true, true, false, false, 3},
        {"Attention backend disabled", generous_heap, 1, 0, true, false, true, false, false, 0},
        {"Attention optimization disabled", generous_heap, 1, 0, true, true, false, false, false, 0},
        {"zero sessions uses one", generous_heap, 0, 0, true, true, true, false, false, 3},
    }};

    for (const BudgetCase& test_case : cases)
    {
        CompilerOption opt;
        opt.flags |= BackendVulkanDense | BackendFileBackedExperts;
        if (test_case.gpu_experts)
            opt.flags |= BackendVulkanExperts;
        if (test_case.attention_backend)
            opt.flags |= BackendVulkanAttention;
        opt.optimization_flags = g_test_optimization_flags & ~OptimizationVulkanCommandGraph;
        if (!test_case.attention_optimization)
            opt.optimization_flags &= ~OptimizationVulkanAttention;
        opt.device_index = get_default_gpu_index();
        opt.vulkan_runtime = create_vulkan_runtime();
        opt.gpu_heap_budget = test_case.heap_size;
        opt.num_concurrent_sessions = test_case.sessions;
        opt.gpu_expert_cache_reserve = test_case.cache_reserve;
        if (test_case.unavailable_snapshot)
            opt.gpu_resources.push_back({opt.device_index, 0, 0});
        CompiledModel model;
        auto compiled = compile_model(descriptor, mapping, model, HybridMode::HybridExperts, opt);
        if (test_case.expect_error)
        {
            check(!compiled && compiled.error().code == ErrorCode::InvalidArgument);
            check(compiled.error().message.find("Vulkan resource plan cannot allocate") != std::string::npos);
            continue;
        }
        if (!compiled)
            throw std::runtime_error(std::string("GDN budget compilation failed (") + test_case.name + "): " + compiled.error().message);
        check(!model.gated_residual_chain);

        uint32_t admitted_layers = 0;
        for (const CompiledLayerPlan& layer : model.net.layer_plans)
        {
            const AttentionBlockPlan& attention = layer.attention;
            const bool admitted = static_cast<bool>(model.operators.at(attention.gated_delta_vulkan_operator).gated_delta);
            admitted_layers += admitted;
            check((attention.fused_delta_input_operator != invalid_compiled_operator_handle) == admitted);
            check(static_cast<bool>(model.operators.at_weight(attention.output_weight).bfloat16) == admitted);
            const auto node = std::find_if(model.net.layers.begin(), model.net.layers.end(), [&](const Layer& candidate) {
                return candidate.kind == LayerKind::Attention && candidate.plan_index == layer.layer_id;
            });
            check(node != model.net.layers.end());
            check((node->backend == ExecutionBackend::Vulkan) == admitted);
            for (const ExpertPlan& expert : layer.moe.experts)
                check(static_cast<bool>(model.weights.at(expert.gate_up_weight).mxfp4_file_storage));
        }
        if (admitted_layers != (bf16_supported ? test_case.expected_layers : 0))
            throw std::runtime_error(std::string("unexpected GDN budget admission (") + test_case.name + "): " + std::to_string(admitted_layers));
        for (const GpuResourcePlan& resources : model.memory_plan.gpu_devices)
        {
            if (resources.admitted.persistent_size != 0)
                check(can_admit_gpu_resources(resources, resources.initial, {}));
            check(resources.expert_cache_reserve >= test_case.cache_reserve);
        }
        uint32_t gpu_expert_nodes = 0;
        for (const Layer& node : model.net.layers)
            if (node.kind == LayerKind::ExpertGroup && node.backend == ExecutionBackend::Vulkan)
                ++gpu_expert_nodes;
        check(gpu_expert_nodes == (test_case.gpu_experts ? descriptor.layers.size() : 0));

        std::array<LayerCache, 3> cpu_caches;
        std::array<LayerCache, 3> gpu_caches;
        std::array<GatedDeltaScratch, 3> cpu_scratches;
        std::array<GatedDeltaScratch, 3> gpu_scratches;
        const VulkanStatistics before = get_vulkan_statistics(model.vulkan_runtime);
        bool nonzero_output = false;
        constexpr uint32_t token_count = 3;
        for (uint32_t token = 0; token < token_count; ++token)
        {
            for (uint32_t layer_id = 0; layer_id < descriptor.layers.size(); ++layer_id)
            {
                ActivationBuffer input(1, descriptor.hidden_size);
                for (uint32_t column = 0; column < input.columns(); ++column)
                    input.row(0)[column] = static_cast<float>(static_cast<int>((column * 5 + token * 3 + layer_id * 7) % 19) - 9) / 16.0f;
                ActivationBuffer expected;
                ActivationBuffer actual;
                check(static_cast<bool>(forward_gated_delta(cpu_model.weights, cpu_model.operators, cpu_model.net.layer_plans[layer_id].attention,
                                                            ExecutionBackend::Cpu, cpu_caches[layer_id], cpu_scratches[layer_id], input, expected,
                                                            cpu_model.opt.optimization_flags)));
                const AttentionBlockPlan& selected_attention = model.net.layer_plans[layer_id].attention;
                const ExecutionBackend selected_backend = model.operators.at(selected_attention.gated_delta_vulkan_operator).gated_delta
                                                              ? ExecutionBackend::Vulkan
                                                              : ExecutionBackend::Cpu;
                check(static_cast<bool>(forward_gated_delta(model.weights, model.operators, selected_attention,
                                                            selected_backend, gpu_caches[layer_id], gpu_scratches[layer_id], input, actual,
                                                            model.opt.optimization_flags)));
                check(actual.rows() == expected.rows() && actual.columns() == expected.columns());
                for (uint32_t column = 0; column < actual.columns(); ++column)
                {
                    const float reference = expected.row(0)[column];
                    nonzero_output = nonzero_output || std::abs(reference) > 1e-5f;
                    check_near(actual.row(0)[column], reference, 2e-4f + 2e-4f * std::abs(reference));
                }
                check(gpu_caches[layer_id].gated_delta_token_count == token + 1);
                check(static_cast<bool>(gpu_caches[layer_id].gated_delta_device_state) == (layer_id < admitted_layers));
                check((gpu_caches[layer_id].device_allocated_size != 0) == (layer_id < admitted_layers));
            }
        }
        check(nonzero_output);
        const VulkanStatistics after = get_vulkan_statistics(model.vulkan_runtime);
        check(after.gated_delta_fusions - before.gated_delta_fusions == admitted_layers * token_count);
        check(after.gated_delta_submissions - before.gated_delta_submissions == admitted_layers * token_count);
        if (admitted_layers != 0)
            check(after.dispatches > before.dispatches && after.compute_submissions > before.compute_submissions);
        else
            check(after.compute_submissions == before.compute_submissions);

        for (uint32_t layer_id = 0; layer_id < admitted_layers; ++layer_id)
        {
            std::vector<float> convolution;
            std::vector<float> recurrent;
            const VulkanStatistics download_before = get_vulkan_statistics(model.vulkan_runtime);
            if (!gpu_caches[layer_id].gated_delta_device_state->download(convolution, recurrent))
                throw std::runtime_error(std::string("GDN state download failed (") + test_case.name + "), layer " + std::to_string(layer_id));
            const VulkanStatistics download_after = get_vulkan_statistics(model.vulkan_runtime);
            check(download_after.compute_submissions == download_before.compute_submissions + 1);
            check(download_after.batch_downloads == download_before.batch_downloads + 1);
            check(download_after.dispatches == download_before.dispatches);
            check(download_after.gated_delta_fusions == download_before.gated_delta_fusions);
            check(convolution.size() == cpu_caches[layer_id].gated_delta_convolution.size());
            check(recurrent.size() == cpu_caches[layer_id].gated_delta_recurrent.size());
            for (size_t index = 0; index < convolution.size(); ++index)
                check_near(convolution[index], cpu_caches[layer_id].gated_delta_convolution[index], 2e-5f);
            for (size_t index = 0; index < recurrent.size(); ++index)
                check_near(recurrent[index], cpu_caches[layer_id].gated_delta_recurrent[index], 2e-5f);
            std::vector<float> repeated_convolution;
            std::vector<float> repeated_recurrent;
            if (!gpu_caches[layer_id].gated_delta_device_state->download(repeated_convolution, repeated_recurrent))
                throw std::runtime_error(std::string("repeated GDN state download failed (") + test_case.name + "), layer " + std::to_string(layer_id));
            check(repeated_convolution == convolution);
            check(repeated_recurrent == recurrent);
        }
    }
}

void test_gated_residual_kernels()
{
    WeightStore weights;
    const TensorHandle norm = add_bfloat16_tensor(weights, "gr_norm", {4}, {0.0f, 0.0f, 0.0f, 0.0f});
    const TensorHandle mix_down = add_bfloat16_tensor(weights, "gr_down", {1, 4}, {0.0f, 0.0f, 0.0f, 0.0f});
    const TensorHandle mix_up = add_bfloat16_tensor(weights, "gr_up", {4, 1}, {0.0f, 0.0f, 0.0f, 0.0f});
    const TensorHandle inject = add_bfloat16_tensor(weights, "gr_inject", {2, 4},
                                                    {0.0f, 0.0f, 0.0f, 0.0f,
                                                     0.0f, 0.0f, 0.0f, 0.0f});
    CompiledOperatorTable operators;
    MoeModelDescriptor descriptor;
    descriptor.hyper_connection_multiplier = 2;
    descriptor.hidden_size = 2;
    descriptor.norm_epsilon = 1e-6f;
    descriptor.norm_weight_offset = 1.0f;
    GatedResidualPlan plan;
    plan.norm_weight = norm;
    plan.mix_down_weight = mix_down;
    plan.mix_up_weight = mix_up;
    plan.inject_weight = inject;

    ActivationBuffer input(1, 4);
    input.row(0)[0] = 1.0f;
    input.row(0)[1] = 2.0f;
    input.row(0)[2] = 3.0f;
    input.row(0)[3] = 4.0f;
    constexpr float epsilon = 1e-6f;
    HyperConnectionMix mixed;
    HyperConnectionScratch gated_scratch;
    auto mixed_result = gr_pre(input,
                               weights,
                               operators,
                               plan,
                               descriptor,
                               mixed,
                               gated_scratch,
                               0);
    check(static_cast<bool>(mixed_result));
    const float first_scale = 1.0f / std::sqrt(2.5f + epsilon);
    const float second_scale = 1.0f / std::sqrt(12.5f + epsilon);
    check_near(mixed.reduced.row(0)[0],
               (first_scale + 3.0f * second_scale) * 0.25f,
               1e-6f);
    check_near(mixed.reduced.row(0)[1],
               (2.0f * first_scale + 4.0f * second_scale) * 0.25f,
               1e-6f);
    check(mixed.post.size() == 2);
    check_near(mixed.post[0], 1.0f, 1e-6f);
    check_near(mixed.post[1], 1.0f, 1e-6f);

    ActivationBuffer larger_input(3, 4);
    for (size_t row = 0; row < larger_input.rows(); ++row)
        std::copy_n(input.row(0), input.columns(), larger_input.row(row));
    auto enlarged = gr_pre(larger_input,
                           weights,
                           operators,
                           plan,
                           descriptor,
                           mixed,
                           gated_scratch,
                           0);
    check(static_cast<bool>(enlarged));
    const uint64_t grown_reduced_capacity = mixed.reduced.allocated_bytes();
    const uint64_t grown_normalized_capacity = gated_scratch.normalized.allocated_bytes();
    const std::byte* grown_reduced_data = mixed.reduced.bytes().data();
    const std::byte* grown_normalized_data = gated_scratch.normalized.bytes().data();
    auto restored = gr_pre(input,
                           weights,
                           operators,
                           plan,
                           descriptor,
                           mixed,
                           gated_scratch,
                           0);
    check(static_cast<bool>(restored));
    check(mixed.reduced.allocated_bytes() == grown_reduced_capacity);
    check(gated_scratch.normalized.allocated_bytes() == grown_normalized_capacity);
    check(mixed.reduced.bytes().data() == grown_reduced_data);
    check(gated_scratch.normalized.bytes().data() == grown_normalized_data);

    const auto add_zero_quantized_projection = [&weights](const char* name,
                                                          std::vector<uint32_t> shape,
                                                          DType dtype) {
        TensorData tensor;
        tensor.dtype = dtype;
        tensor.shape = std::move(shape);
        if (dtype == DType::Int8)
        {
            tensor.int8_data.assign(static_cast<size_t>(tensor.shape[0]) * tensor.shape[1], 0);
            tensor.quantization_scales.assign(tensor.shape[0], 1.0f);
        }
        else
        {
            tensor.quantized_data.assign(static_cast<size_t>(tensor.shape[0]) * tensor.shape[1], 0);
            tensor.quantization_scales = {1.0f};
        }
        auto added = weights.add(name, std::move(tensor));
        check(static_cast<bool>(added));
        return added.value();
    };
    GatedResidualPlan int8_plan = plan;
    int8_plan.mix_down_weight = add_zero_quantized_projection("gr_int8_down", {1, 4}, DType::Int8);
    int8_plan.mix_up_weight = add_zero_quantized_projection("gr_int8_up", {4, 1}, DType::Int8);
    int8_plan.inject_weight = add_zero_quantized_projection("gr_int8_inject", {2, 4}, DType::Int8);
    HyperConnectionMix int8_mixed;
    HyperConnectionScratch int8_scratch;
    auto int8_pre = gr_pre(input,
                           weights,
                           operators,
                           int8_plan,
                           descriptor,
                           int8_mixed,
                           int8_scratch,
                           0);
    check(static_cast<bool>(int8_pre));
    for (size_t index = 0; index < mixed.reduced.values().size(); ++index)
        check_near(int8_mixed.reduced.values()[index], mixed.reduced.values()[index], 1e-6f);
    for (size_t index = 0; index < mixed.post.size(); ++index)
        check_near(int8_mixed.post[index], mixed.post[index], 1e-6f);

    GatedResidualPlan float8_plan = plan;
    float8_plan.mix_down_weight = add_zero_quantized_projection("gr_float8_down", {1, 4}, DType::Float8E4M3);
    float8_plan.mix_up_weight = add_zero_quantized_projection("gr_float8_up", {4, 1}, DType::Float8E4M3);
    float8_plan.inject_weight = add_zero_quantized_projection("gr_float8_inject", {2, 4}, DType::Float8E4M3);
    HyperConnectionMix float8_mixed;
    HyperConnectionScratch float8_scratch;
    auto float8_pre = gr_pre(input,
                             weights,
                             operators,
                             float8_plan,
                             descriptor,
                             float8_mixed,
                             float8_scratch,
                             0);
    check(static_cast<bool>(float8_pre));
    const std::byte* auxiliary_storage = float8_scratch.auxiliary.bytes().data();
    const uint64_t auxiliary_capacity = float8_scratch.auxiliary.allocated_bytes();
    const std::byte* reduced_storage = float8_mixed.reduced.bytes().data();
    const uint64_t reduced_capacity = float8_mixed.reduced.allocated_bytes();
    check(reduced_capacity >= 4 * sizeof(float));
    float8_pre = gr_pre(input,
                        weights,
                        operators,
                        float8_plan,
                        descriptor,
                        float8_mixed,
                        float8_scratch,
                        0);
    check(static_cast<bool>(float8_pre));
    check(float8_scratch.auxiliary.bytes().data() == auxiliary_storage);
    check(float8_scratch.auxiliary.allocated_bytes() == auxiliary_capacity);
    check(float8_mixed.reduced.bytes().data() == reduced_storage);
    check(float8_mixed.reduced.allocated_bytes() == reduced_capacity);
    for (size_t index = 0; index < mixed.reduced.values().size(); ++index)
        check_near(float8_mixed.reduced.values()[index], mixed.reduced.values()[index], 1e-6f);
    for (size_t index = 0; index < mixed.post.size(); ++index)
        check_near(float8_mixed.post[index], mixed.post[index], 1e-6f);

    TensorData malformed_norm = weights.at(norm);
    malformed_norm.shape.clear();
    auto malformed_norm_added = weights.add("gr_malformed_norm", malformed_norm);
    check(static_cast<bool>(malformed_norm_added));
    GatedResidualPlan malformed_norm_plan = plan;
    malformed_norm_plan.norm_weight = malformed_norm_added.value();
    auto invalid_norm = gr_pre(input,
                               weights,
                               operators,
                               malformed_norm_plan,
                               descriptor,
                               mixed,
                               gated_scratch,
                               0);
    check(!invalid_norm);
    check(invalid_norm.error().code == ErrorCode::InvalidModel);
    check(invalid_norm.error().message == "invalid gated-residual normalization tensor");

    weights.at_mutable(malformed_norm_added.value()).shape = {2, 2};
    auto invalid_norm_rank = gr_pre(input,
                                    weights,
                                    operators,
                                    malformed_norm_plan,
                                    descriptor,
                                    mixed,
                                    gated_scratch,
                                    0);
    check(!invalid_norm_rank);
    check(invalid_norm_rank.error().code == ErrorCode::InvalidModel);
    check(invalid_norm_rank.error().message == "invalid gated-residual normalization tensor");

    TensorData malformed_mix_up = weights.at(mix_up);
    malformed_mix_up.shape = {2, 2};
    auto malformed_mix_up_added = weights.add("gr_malformed_mix_up", malformed_mix_up);
    check(static_cast<bool>(malformed_mix_up_added));
    GatedResidualPlan malformed_mix_up_plan = plan;
    malformed_mix_up_plan.mix_up_weight = malformed_mix_up_added.value();
    auto invalid_mix_up = gr_pre(input,
                                 weights,
                                 operators,
                                 malformed_mix_up_plan,
                                 descriptor,
                                 mixed,
                                 gated_scratch,
                                 0);
    check(!invalid_mix_up);
    check(invalid_mix_up.error().code == ErrorCode::InvalidModel);
    check(invalid_mix_up.error().message == "invalid gated-residual up projection");

    ActivationBuffer branch(1, 2);
    branch.row(0)[0] = 10.0f;
    branch.row(0)[1] = 20.0f;
    ActivationBuffer residual = input;
    ActivationBuffer posted_output;
    auto posted = gr_post(branch, residual, mixed, 2, posted_output);
    check(static_cast<bool>(posted));
    check_near(posted_output.row(0)[0], 11.0f, 1e-6f);
    check_near(posted_output.row(0)[1], 22.0f, 1e-6f);
    check_near(posted_output.row(0)[2], 13.0f, 1e-6f);
    check_near(posted_output.row(0)[3], 24.0f, 1e-6f);

    auto aliased_branch = gr_post(branch, residual, mixed, 2, branch);
    check(!aliased_branch);
    check(aliased_branch.error().code == ErrorCode::InvalidArgument);
    check_near(branch.row(0)[0], 10.0f, 1e-6f);
    auto in_place = gr_post(branch, residual, mixed, 2, residual);
    check(static_cast<bool>(in_place));
    check_near(residual.row(0)[0], 11.0f, 1e-6f);
    check_near(residual.row(0)[1], 22.0f, 1e-6f);
    check_near(residual.row(0)[2], 13.0f, 1e-6f);
    check_near(residual.row(0)[3], 24.0f, 1e-6f);

    const std::array<float, 2> expected_head = {
        mixed.reduced.row(0)[0], mixed.reduced.row(0)[1]};
    ActivationBuffer head_output(1, 1);
    head_output.row(0)[0] = 123.0f;
    auto head = gr_head(input,
                        weights,
                        operators,
                        plan,
                        descriptor,
                        head_output,
                        gated_scratch,
                        0);
    check(static_cast<bool>(head));
    check_near(head_output.row(0)[0], expected_head[0], 1e-6f);
    check_near(head_output.row(0)[1], expected_head[1], 1e-6f);
    std::fill_n(head_output.row(0), head_output.columns(), 456.0f);
    GatedResidualPlan head_plan = plan;
    head_plan.inject_weight = invalid_tensor_handle;
    head = gr_head(input,
                   weights,
                   operators,
                   head_plan,
                   descriptor,
                   head_output,
                   gated_scratch,
                   0);
    check(static_cast<bool>(head));
    check_near(head_output.row(0)[0], expected_head[0], 1e-6f);
    check_near(head_output.row(0)[1], expected_head[1], 1e-6f);
    auto aliased_head = gr_head(input,
                                weights,
                                operators,
                                plan,
                                descriptor,
                                input,
                                gated_scratch,
                                0);
    check(!aliased_head);
    check(aliased_head.error().code == ErrorCode::InvalidArgument);
    check_near(input.row(0)[0], 1.0f, 1e-6f);
    check_near(input.row(0)[3], 4.0f, 1e-6f);
}

void test_ple_prefill_decode_continuation()
{
    WeightStore weights;
    CompiledOperatorTable operators;
    MoeModelDescriptor descriptor;
    descriptor.hyper_connection_multiplier = 2;
    descriptor.hidden_size = 2;
    descriptor.norm_epsilon = 1e-6f;
    descriptor.norm_weight_offset = 1.0f;
    PleBlockPlan plan;
    plan.embedding_dimension = 4;
    plan.convolution_kernel_size = 2;
    plan.ngram_size = 3;
    plan.heads_per_ngram = 1;
    plan.eos_token_id = 9;
    plan.hash_multipliers = add_int64_tensor(weights, "ple_multipliers", {3}, {1, 3, 5});
    plan.head_vocabulary_sizes = add_int64_tensor(weights, "ple_vocab", {2}, {2, 2});
    plan.head_offsets = add_int64_tensor(weights, "ple_offsets", {2}, {0, 2});
    plan.embedding_shards.push_back(add_bfloat16_tensor(weights,
                                                        "ple_embedding",
                                                        {4, 2},
                                                        {1.0f, 2.0f,
                                                         3.0f, 4.0f,
                                                         5.0f, 6.0f,
                                                         7.0f, 8.0f}));
    const auto add_float8 = [&weights](const char* name,
                                       std::vector<uint32_t> shape,
                                       std::vector<float> values) {
        TensorData tensor;
        tensor.dtype = DType::Float8E4M3;
        tensor.shape = std::move(shape);
        tensor.quantized_data.reserve(values.size());
        for (float value : values)
            tensor.quantized_data.push_back(float_to_float8_e4m3(value));
        tensor.quantization_scales = {1.0f};
        auto added = weights.add(name, std::move(tensor));
        check(static_cast<bool>(added));
        return added.value();
    };
    plan.key_weight = add_float8("ple_key", {4, 4}, std::vector<float>(16, 0.0f));
    plan.value_weight = add_float8("ple_value",
                                   {2, 4},
                                   {1.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f});
    plan.key_norm_weight = add_bfloat16_tensor(weights, "ple_key_norm", {4}, {0.0f, 0.0f, 0.0f, 0.0f});
    plan.query_norm_weight = add_bfloat16_tensor(weights, "ple_query_norm", {4}, {0.0f, 0.0f, 0.0f, 0.0f});
    plan.convolution_norm_weight = add_bfloat16_tensor(weights, "ple_conv_norm", {4}, {0.0f, 0.0f, 0.0f, 0.0f});
    plan.convolution_weight = add_bfloat16_tensor(weights,
                                                  "ple_conv",
                                                  {4, 1, 2},
                                                  {1.0f, 0.0f,
                                                   1.0f, 0.0f,
                                                   1.0f, 0.0f,
                                                   1.0f, 0.0f});

    const std::array<int32_t, 4> input_ids = {1, 2, 9, 3};
    ActivationBuffer prefill_hidden(input_ids.size(), 4);
    LayerCache prefill_cache;
    AttentionScratch prefill_scratch;
    check(static_cast<bool>(forward_ple(weights,
                                        operators,
                                        plan,
                                        descriptor,
                                        input_ids,
                                        prefill_cache,
                                        prefill_scratch,
                                        prefill_hidden,
                                        0)));

    ActivationBuffer decode_hidden(input_ids.size(), 4);
    LayerCache decode_cache;
    const std::array<const std::byte*, 8> storage = {
        prefill_scratch.normalized.bytes().data(),
        prefill_scratch.key.bytes().data(),
        prefill_scratch.value.bytes().data(),
        prefill_scratch.query.bytes().data(),
        prefill_scratch.gate.bytes().data(),
        prefill_scratch.attention.bytes().data(),
        prefill_scratch.projected.bytes().data(),
        prefill_scratch.quantized_input.bytes().data(),
    };
    AttentionScratch decode_scratch = std::move(prefill_scratch);
    for (size_t row = 0; row < input_ids.size(); ++row)
    {
        ActivationBuffer token_hidden(1, 4);
        const std::array<int32_t, 1> token = {input_ids[row]};
        check(static_cast<bool>(forward_ple(weights,
                                            operators,
                                            plan,
                                            descriptor,
                                            token,
                                            decode_cache,
                                            decode_scratch,
                                            token_hidden,
                                            0)));
        std::copy_n(token_hidden.row(0), 4, decode_hidden.row(row));
    }
    for (size_t row = 0; row < input_ids.size(); ++row)
        for (size_t column = 0; column < 4; ++column)
            check_near(prefill_hidden.row(row)[column], decode_hidden.row(row)[column], 1e-6f);
    check(decode_scratch.normalized.bytes().data() == storage[0]);
    check(decode_scratch.key.bytes().data() == storage[1]);
    check(decode_scratch.value.bytes().data() == storage[2]);
    check(decode_scratch.query.bytes().data() == storage[3]);
    check(decode_scratch.gate.bytes().data() == storage[4]);
    check(decode_scratch.attention.bytes().data() == storage[5]);
    check(decode_scratch.projected.bytes().data() == storage[6]);
    check(decode_scratch.quantized_input.bytes().data() == storage[7]);
    check(decode_scratch.quantized_input.allocated_bytes() >= 4 * sizeof(float));
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
        check_near(prefill_cache.ple_convolution_state[index],
                   decode_cache.ple_convolution_state[index],
                   1e-6f);
    }

    // Compare the ring against a history kept in canonical oldest-first order.
    for (uint32_t kernel_size : {1u, 2u, 4u})
    {
        PleBlockPlan ring_plan = plan;
        ring_plan.convolution_kernel_size = kernel_size;
        std::vector<float> convolution_weight(4 * kernel_size);
        for (uint32_t channel = 0; channel < 4; ++channel)
            for (uint32_t kernel = 0; kernel < kernel_size; ++kernel)
                convolution_weight[channel * kernel_size + kernel] = 0.125f * static_cast<float>(channel + kernel + 1);
        ring_plan.convolution_weight = add_bfloat16_tensor(weights, "ple_ring_conv_" + std::to_string(kernel_size),
                                                           {4, 1, kernel_size}, convolution_weight);
        const std::array<int32_t, 23> tokens = {1, 2, 9, 3, 1, 2, 3, 1, 9, 2, 1, 3, 1, 2, 3, 9, 1, 2, 1, 3, 2, 1, 2};
        LayerCache ring_cache;
        LayerCache canonical_cache;
        LayerCache batch_cache;
        AttentionScratch ring_scratch;
        AttentionScratch canonical_scratch;
        AttentionScratch batch_scratch;
        ActivationBuffer batch_hidden(tokens.size(), 4);
        check(static_cast<bool>(forward_ple(weights, operators, ring_plan, descriptor,
                                            tokens, batch_cache, batch_scratch, batch_hidden, 0)));
        for (size_t row = 0; row < tokens.size(); ++row)
        {
            const std::span<const int32_t> token(tokens.data() + row, 1);
            ActivationBuffer ring_hidden(1, 4);
            ActivationBuffer canonical_hidden(1, 4);
            check(static_cast<bool>(forward_ple(weights, operators, ring_plan, descriptor,
                                                token, ring_cache, ring_scratch, ring_hidden, 0)));
            check(static_cast<bool>(forward_ple(weights, operators, ring_plan, descriptor,
                                                token, canonical_cache, canonical_scratch, canonical_hidden, 0)));
            for (size_t column = 0; column < 4; ++column)
            {
                check_near(ring_hidden.row(0)[column], canonical_hidden.row(0)[column], 1e-6f);
                check_near(ring_hidden.row(0)[column], batch_hidden.row(row)[column], 1e-6f);
            }
            std::rotate(canonical_cache.ple_convolution_state.begin(),
                        canonical_cache.ple_convolution_state.begin() + canonical_cache.ple_first_slot * 4,
                        canonical_cache.ple_convolution_state.end());
            canonical_cache.ple_first_slot = 0;
            std::vector<float> canonical_ring = ring_cache.ple_convolution_state;
            std::rotate(canonical_ring.begin(), canonical_ring.begin() + ring_cache.ple_first_slot * 4, canonical_ring.end());
            check(canonical_ring == canonical_cache.ple_convolution_state);
            check(ring_cache.ple_token_history == canonical_cache.ple_token_history);
        }
        check(ring_cache.ple_convolution_state == batch_cache.ple_convolution_state);
        check(ring_cache.ple_first_slot == batch_cache.ple_first_slot);
        ring_cache = {};
        ActivationBuffer reset_hidden(1, 4);
        check(static_cast<bool>(forward_ple(weights, operators, ring_plan, descriptor,
                                            std::span<const int32_t>(tokens.data(), 1), ring_cache, ring_scratch, reset_hidden, 0)));
        for (size_t column = 0; column < 4; ++column)
            check_near(reset_hidden.row(0)[column], batch_hidden.row(0)[column], 1e-6f);
    }

    {
        WeightStore hash_weights;
        PleBlockPlan hash_plan;
        hash_plan.embedding_dimension = 4;
        hash_plan.convolution_kernel_size = 1;
        hash_plan.ngram_size = 3;
        hash_plan.heads_per_ngram = 1;
        hash_plan.eos_token_id = 9;
        hash_plan.hash_multipliers = add_int64_tensor(hash_weights,
                                                      "ple_hash_regression_multipliers",
                                                      {3},
                                                      {1, std::numeric_limits<int64_t>::min(),
                                                       std::numeric_limits<int64_t>::min()});
        hash_plan.head_vocabulary_sizes = add_int64_tensor(hash_weights, "ple_hash_regression_vocab", {2}, {5, 5});
        hash_plan.head_offsets = add_int64_tensor(hash_weights, "ple_hash_regression_offsets", {2}, {0, 5});
        hash_plan.embedding_shards.push_back(add_bfloat16_tensor(hash_weights,
                                                                 "ple_hash_regression_embedding",
                                                                 {10, 2},
                                                                 {1.0f, 2.0f,
                                                                  11.0f, 12.0f,
                                                                  21.0f, 22.0f,
                                                                  31.0f, 32.0f,
                                                                  41.0f, 42.0f,
                                                                  51.0f, 52.0f,
                                                                  61.0f, 62.0f,
                                                                  71.0f, 72.0f,
                                                                  81.0f, 82.0f,
                                                                  91.0f, 92.0f}));
        hash_plan.key_weight = add_bfloat16_tensor(hash_weights,
                                                   "ple_hash_regression_key",
                                                   {4, 4},
                                                   std::vector<float>(16, 0.0f));
        hash_plan.value_weight = add_bfloat16_tensor(hash_weights,
                                                     "ple_hash_regression_value",
                                                     {4, 4},
                                                     {1.0f, 0.0f, 0.0f, 0.0f,
                                                      0.0f, 1.0f, 0.0f, 0.0f,
                                                      0.0f, 0.0f, 1.0f, 0.0f,
                                                      0.0f, 0.0f, 0.0f, 1.0f});
        hash_plan.key_norm_weight = add_bfloat16_tensor(hash_weights, "ple_hash_regression_key_norm", {4}, std::vector<float>(4, 0.0f));
        hash_plan.query_norm_weight = add_bfloat16_tensor(hash_weights, "ple_hash_regression_query_norm", {4}, std::vector<float>(4, 0.0f));
        hash_plan.convolution_norm_weight = add_bfloat16_tensor(hash_weights, "ple_hash_regression_conv_norm", {4}, std::vector<float>(4, 0.0f));
        hash_plan.convolution_weight = add_bfloat16_tensor(hash_weights,
                                                           "ple_hash_regression_conv",
                                                           {4, 1, 1},
                                                           std::vector<float>(4, 0.0f));

        const std::array<int32_t, 5> hash_input_ids = {-2, 7, 2, 9, 3};
        ActivationBuffer hash_hidden(hash_input_ids.size(), 4);
        LayerCache hash_cache;
        AttentionScratch hash_scratch;
        CompiledOperatorTable hash_operators;
        MoeModelDescriptor hash_descriptor;
        hash_descriptor.hyper_connection_multiplier = 1;
        hash_descriptor.hidden_size = 4;
        hash_descriptor.norm_epsilon = 1e-6f;
        hash_descriptor.norm_weight_offset = 1.0f;
        check(static_cast<bool>(forward_ple(hash_weights,
                                            hash_operators,
                                            hash_plan,
                                            hash_descriptor,
                                            hash_input_ids,
                                            hash_cache,
                                            hash_scratch,
                                            hash_hidden,
                                            0)));

        // Golden rows are embedding heads through identity values and a 0.5 gate.
        const std::array<std::array<float, 4>, 5> expected_hash_outputs = {{
            {5.5f, 6.0f, 40.5f, 41.0f},
            {10.5f, 11.0f, 45.5f, 46.0f},
            {20.5f, 21.0f, 45.5f, 46.0f},
            {20.5f, 21.0f, 30.5f, 31.0f},
            {0.5f, 1.0f, 40.5f, 41.0f},
        }};
        for (size_t row = 0; row < expected_hash_outputs.size(); ++row)
            for (size_t column = 0; column < expected_hash_outputs[row].size(); ++column)
                check_near(hash_hidden.row(row)[column], expected_hash_outputs[row][column], 1e-6f);
    }
}

void test_quantized_attention_scratch_reuse()
{
    constexpr uint32_t input_columns = 256;
    const uint64_t flags = g_test_optimization_flags & ~OptimizationCpuFloat8FusedProjections
                           & ~OptimizationCpuPackedWeights;
    for (DType dtype : {DType::Float8E4M3, DType::Int4, DType::Q8K})
        for (bool delta : {false, true})
        {
            WeightStore weights;
            CompiledOperatorTable operators;
            const auto add_projection = [&](uint32_t rows, uint32_t seed) {
                TensorData matrix;
                matrix.dtype = dtype;
                matrix.shape = {rows, input_columns};
                std::vector<float> decoded(static_cast<size_t>(rows) * input_columns);
                std::vector<uint8_t> raw;
                if (dtype == DType::Float8E4M3)
                {
                    raw.resize(decoded.size());
                    matrix.float8_scale_row_size = 1;
                    matrix.quantization_scales.assign(static_cast<size_t>(rows) * 2, 1.0f);
                    for (size_t index = 0; index < raw.size(); ++index)
                    {
                        raw[index] = float_to_float8_e4m3(static_cast<float>(static_cast<int>((index * 3 + seed) % 16) - 8) / 64.0f);
                        decoded[index] = float8_e4m3_to_float(raw[index]);
                    }
                }
                else
                {
                    matrix.input_column_indices.resize(input_columns);
                    for (uint32_t column = 0; column < input_columns; ++column)
                        matrix.input_column_indices[column] = input_columns - column - 1;
                    if (dtype == DType::Int4)
                    {
                        matrix.quantization_group_size = 32;
                        matrix.quantization_scales.assign(static_cast<size_t>(rows) * 8, 1.0f / 64.0f);
                        matrix.quantization_zero_points.assign(matrix.quantization_scales.size(), 8);
                        raw.resize(static_cast<size_t>(rows) * input_columns / 2);
                        for (size_t index = 0; index < raw.size(); ++index)
                            raw[index] = static_cast<uint8_t>(((index * 3 + seed) % 16)
                                                              | (((index * 5 + seed + 1) % 16) << 4));
                    }
                    else
                    {
                        raw.resize(static_cast<size_t>(qnk_storage_bytes(dtype, rows, input_columns)));
                        std::array<float, input_columns> source;
                        std::array<float, input_columns> block;
                        for (uint32_t row = 0; row < rows; ++row)
                        {
                            for (uint32_t column = 0; column < input_columns; ++column)
                                source[column] = static_cast<float>(static_cast<int>((column * 3 + row + seed) % 16) - 8) / 64.0f;
                            uint8_t* encoded = raw.data() + row * qnk_block_bytes(dtype);
                            qnk_q8k_quantize(source.data(), encoded, input_columns);
                            qnk_dequantize_block(dtype, encoded, block.data());
                            for (uint32_t column = 0; column < input_columns; ++column)
                                decoded[static_cast<size_t>(row) * input_columns + matrix.input_column_indices[column]] = block[column];
                        }
                    }
                }
                std::shared_ptr<uint8_t[]> storage(new uint8_t[raw.size()], std::default_delete<uint8_t[]>());
                std::copy(raw.begin(), raw.end(), storage.get());
                matrix.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
                matrix.mapped_size = raw.size();
                if (dtype == DType::Int4)
                    for (uint32_t row = 0; row < rows; ++row)
                        check(int4_dequantize_row(matrix, row, decoded.data() + static_cast<size_t>(row) * input_columns));
                const std::string name = "scratch_projection_" + std::to_string(weights.size());
                const auto added = weights.add(name, std::move(matrix));
                check(static_cast<bool>(added));
                const TensorHandle reference = add_float_tensor(weights, name + "_reference", {rows, input_columns}, std::move(decoded));
                return std::pair<TensorHandle, TensorHandle>{added.value(), reference};
            };

            AttentionBlockPlan plan;
            plan.kind = delta ? AttentionKind::GatedDeltaNet : AttentionKind::Standard;
            plan.flags = AttentionBlockExternalResidual;
            plan.head_count = 1;
            plan.kv_head_count = 1;
            plan.head_dimension = 2;
            plan.value_head_dimension = 2;
            plan.rope_head_dimension = 2;
            plan.norm_epsilon = 1e-6f;
            std::vector<float> output_weights(input_columns * 2);
            for (size_t index = 0; index < output_weights.size(); ++index)
                output_weights[index] = static_cast<float>(static_cast<int>(index % 7) - 3) / 16.0f;
            plan.output_weight = add_float_tensor(weights, "scratch_output", {input_columns, 2}, std::move(output_weights));
            if (delta)
            {
                plan.convolution_kernel_size = 2;
                plan.delta_convolution_weight = add_float_tensor(weights, "scratch_conv", {6, 1, 2},
                                                                 {0.2f, 0.8f, -0.1f, 0.7f, 0.3f, 0.6f,
                                                                  0.15f, 0.9f, -0.25f, 0.5f, 0.4f, 0.65f});
                plan.delta_time_bias = add_float_tensor(weights, "scratch_time", {1}, {0.1f});
                plan.delta_decay_log = add_float_tensor(weights, "scratch_decay", {1}, {-0.2f});
                plan.delta_norm_weight = add_float_tensor(weights, "scratch_norm", {2}, {1.1f, 0.9f});
            }
            else
            {
                // Biases force the scalar projection path even when fusion is enabled elsewhere.
                plan.query_bias = add_float_tensor(weights, "scratch_q_bias", {2}, {0.02f, -0.03f});
                plan.key_bias = add_float_tensor(weights, "scratch_k_bias", {2}, {-0.01f, 0.04f});
                plan.value_bias = add_float_tensor(weights, "scratch_v_bias", {2}, {0.05f, -0.02f});
            }
            AttentionBlockPlan reference_plan = plan;
            const auto bind_projection = [&](TensorHandle& actual, TensorHandle& reference, uint32_t rows, uint32_t seed) {
                const auto pair = add_projection(rows, seed);
                actual = pair.first;
                reference = pair.second;
            };
            if (delta)
            {
                bind_projection(plan.delta_qkv_weight, reference_plan.delta_qkv_weight, 6, 1);
                bind_projection(plan.delta_z_weight, reference_plan.delta_z_weight, 2, 2);
                bind_projection(plan.delta_beta_weight, reference_plan.delta_beta_weight, 1, 3);
                bind_projection(plan.delta_alpha_weight, reference_plan.delta_alpha_weight, 1, 4);
            }
            else
            {
                bind_projection(plan.query_weight, reference_plan.query_weight, 2, 1);
                bind_projection(plan.key_weight, reference_plan.key_weight, 2, 2);
                bind_projection(plan.value_weight, reference_plan.value_weight, 2, 3);
            }

            for (uint32_t alias = 0; alias < 4; ++alias)
            {
                LayerCache cache;
                LayerCache reference_cache;
                AttentionScratch scratch;
                AttentionScratch reference_scratch;
                GatedDeltaScratch delta_scratch;
                GatedDeltaScratch delta_reference_scratch;
                ActivationBuffer& reused = delta ? delta_scratch.fused_input : scratch.quantized_input;
                if (alias == 3)
                    reused = ActivationBuffer(4, input_columns, DType::BFloat16);
                const std::byte* storage = nullptr;
                uint64_t capacity = 0;
                uint64_t position = 0;
                for (size_t rows : {4u, 1u, 3u, 1u, 4u})
                {
                    ActivationBuffer input(rows, input_columns);
                    for (size_t row = 0; row < rows; ++row)
                        for (uint32_t column = 0; column < input_columns; ++column)
                            input.row(row)[column] = static_cast<float>(static_cast<int>((column * 7 + position + row * 3) % 29) - 14) / 128.0f;
                    ActivationBuffer reference_input = input;
                    if (dtype == DType::Float8E4M3)
                        for (size_t row = 0; row < rows; ++row)
                            quantize_float8_e4m3(input.row(row), reference_input.row(row), input_columns, 128, true, flags);
                    ActivationBuffer expected;
                    ActivationBuffer separate_output;
                    ActivationBuffer& actual = alias == 1   ? input
                                               : alias == 2 ? (delta ? delta_scratch.recurrent_output : scratch.quantized_input)
                                                            : separate_output;
                    std::fill(reused.mutable_bytes().begin(), reused.mutable_bytes().end(), std::byte{0xff});
                    if (delta)
                    {
                        check(static_cast<bool>(forward_gated_delta(weights, operators, reference_plan, ExecutionBackend::Cpu,
                                                                    reference_cache, delta_reference_scratch, reference_input, expected, flags)));
                        check(static_cast<bool>(forward_gated_delta(weights, operators, plan, ExecutionBackend::Cpu,
                                                                    cache, delta_scratch, input, actual, flags)));
                    }
                    else
                    {
                        check(static_cast<bool>(forward_attention(weights, operators, reference_plan, ExecutionBackend::Cpu, position,
                                                                  reference_cache, reference_scratch, reference_input, expected, flags)));
                        check(static_cast<bool>(forward_attention(weights, operators, plan, ExecutionBackend::Cpu, position,
                                                                  cache, scratch, input, actual, flags)));
                    }
                    check(actual.rows() == expected.rows() && actual.columns() == expected.columns());
                    for (size_t index = 0; index < expected.values().size(); ++index)
                    {
                        check(std::isfinite(actual.values()[index]));
                        check_near(actual.values()[index], expected.values()[index], 2e-5f);
                    }
                    check(reused.dtype() == (alias == 3 ? DType::BFloat16 : DType::Float32));
                    if (dtype != DType::Float8E4M3 || &actual == &reused)
                    {
                        check(reused.rows() == (alias == 3 ? 4u : rows) && reused.columns() == input_columns);
                        if (!storage)
                        {
                            storage = reused.bytes().data();
                            capacity = reused.allocated_bytes();
                            check(capacity >= 4 * input_columns * reused.element_size());
                        }
                        else
                        {
                            check(reused.bytes().data() == storage);
                            check(reused.allocated_bytes() == capacity);
                        }
                    }
                    position += rows;
                }
            }

            plan.flags = 0;
            plan.pre_attention_norm_weight = add_float_tensor(weights, "scratch_input_norm", {input_columns},
                                                              std::vector<float>(input_columns, 1.0f));
            LayerCache cache;
            LayerCache reference_cache;
            AttentionScratch scratch;
            GatedDeltaScratch delta_scratch;
            ActivationBuffer& reused = delta ? delta_scratch.fused_input : scratch.quantized_input;
            uint64_t position = 0;
            for (size_t rows : {4u, 1u, 3u})
            {
                ActivationBuffer input(rows, input_columns);
                for (size_t row = 0; row < rows; ++row)
                    for (uint32_t column = 0; column < input_columns; ++column)
                        input.row(row)[column] = static_cast<float>(static_cast<int>((column * 7 + position + row * 3) % 29) - 14) / 128.0f;
                reused = input;
                ActivationBuffer expected;
                ActivationBuffer actual;
                AttentionScratch reference_scratch;
                GatedDeltaScratch delta_reference_scratch;
                if (delta)
                {
                    check(static_cast<bool>(forward_gated_delta(weights, operators, plan, ExecutionBackend::Cpu,
                                                                reference_cache, delta_reference_scratch, input, expected, flags)));
                    check(static_cast<bool>(forward_gated_delta(weights, operators, plan, ExecutionBackend::Cpu,
                                                                cache, delta_scratch, reused, actual, flags)));
                    check(cache.gated_delta_convolution == reference_cache.gated_delta_convolution);
                    check(cache.gated_delta_recurrent == reference_cache.gated_delta_recurrent);
                }
                else
                {
                    check(static_cast<bool>(forward_attention(weights, operators, plan, ExecutionBackend::Cpu, position,
                                                              reference_cache, reference_scratch, input, expected, flags)));
                    check(static_cast<bool>(forward_attention(weights, operators, plan, ExecutionBackend::Cpu, position,
                                                              cache, scratch, reused, actual, flags)));
                    check(cache.keys == reference_cache.keys && cache.values == reference_cache.values);
                    check(cache.bfloat16_keys == reference_cache.bfloat16_keys && cache.bfloat16_values == reference_cache.bfloat16_values);
                }
                check(actual.rows() == expected.rows() && actual.columns() == expected.columns());
                for (size_t index = 0; index < expected.values().size(); ++index)
                {
                    check(std::isfinite(actual.values()[index]));
                    check_near(actual.values()[index], expected.values()[index], 2e-5f);
                }
                check(reused.rows() == input.rows() && reused.columns() == input.columns());
                check(std::equal(reused.values().begin(), reused.values().end(), input.values().begin()));
                position += rows;
            }
        }
}

void test_attention_output_aliases()
{
    WeightStore weights;
    CompiledOperatorTable operators;
    AttentionBlockPlan plan;
    plan.norm_epsilon = 1e-6f;
    plan.kind = AttentionKind::Standard;
    plan.head_count = 1;
    plan.kv_head_count = 1;
    plan.head_dimension = 2;
    plan.value_head_dimension = 2;
    plan.rope_head_dimension = 2;
    const TensorHandle norm = add_float_tensor(weights, "alias_norm", {2}, {0.8f, 1.2f});
    plan.query_weight = add_float_tensor(weights, "alias_query", {2, 2}, {0.5f, -0.2f, 0.3f, 0.7f});
    plan.key_weight = add_float_tensor(weights, "alias_key", {2, 2}, {0.2f, 0.7f, -0.3f, 0.5f});
    plan.value_weight = add_float_tensor(weights, "alias_value", {2, 2}, {0.8f, 0.2f, 0.1f, -0.6f});
    plan.output_weight = add_float_tensor(weights, "alias_output", {2, 2}, {0.9f, -0.2f, 0.3f, 0.4f});
    for (bool normalize : {false, true})
        for (bool external_residual : {false, true})
            for (DType dtype : {DType::Float32, DType::BFloat16})
                for (int alias = 0; alias < 3; ++alias)
                {
                    plan.pre_attention_norm_weight = normalize ? norm : invalid_tensor_handle;
                    plan.kv_cache_dtype = dtype;
                    plan.flags = external_residual ? AttentionBlockExternalResidual : 0;
                    LayerCache reference_cache;
                    LayerCache alias_cache;
                    AttentionScratch reference_scratch;
                    AttentionScratch alias_scratch;
                    ActivationBuffer expected;
                    uint64_t position = 0;
                    for (size_t rows : {3u, 1u, 2u})
                    {
                        ActivationBuffer input(rows, 2);
                        for (size_t row = 0; row < rows; ++row)
                        {
                            input.row(row)[0] = 0.15f * static_cast<float>(position + row + 1);
                            input.row(row)[1] = -0.3f;
                        }
                        ActivationBuffer aliased_input = input;
                        ActivationBuffer& actual = alias == 0   ? aliased_input
                                                   : alias == 1 ? alias_scratch.attention
                                                                : alias_scratch.projected;
                        check(static_cast<bool>(forward_attention(weights, operators, plan, ExecutionBackend::Cpu, position, reference_cache, reference_scratch,
                                                                  input, expected, g_test_optimization_flags)));
                        check(static_cast<bool>(forward_attention(weights, operators, plan, ExecutionBackend::Cpu, position, alias_cache, alias_scratch,
                                                                  aliased_input, actual, g_test_optimization_flags)));
                        check(actual.rows() == expected.rows());
                        check(actual.columns() == expected.columns());
                        for (size_t index = 0; index < expected.values().size(); ++index)
                            check_near(actual.values()[index], expected.values()[index], 1e-6f);
                        check(alias_cache.token_count == reference_cache.token_count);
                        check(alias_cache.keys == reference_cache.keys);
                        check(alias_cache.values == reference_cache.values);
                        check(alias_cache.bfloat16_keys == reference_cache.bfloat16_keys);
                        check(alias_cache.bfloat16_values == reference_cache.bfloat16_values);
                        position += rows;
                    }
                }
    plan.pre_attention_norm_weight = invalid_tensor_handle;
    plan.kv_cache_dtype = DType::Float32;
    plan.flags = 0;
    constexpr uint64_t capacity = 16;
    for (uint32_t window : {0u, 5u})
        for (uint64_t context : {0u, 3u})
            for (size_t rows : {1u, 6u})
            {
                plan.sliding_window = window;
                ActivationBuffer input(rows, 2);
                for (size_t row = 0; row < rows; ++row)
                {
                    input.row(row)[0] = 0.125f * static_cast<float>(row + 1);
                    input.row(row)[1] = -0.0625f * static_cast<float>(row + 2);
                }
                ActivationBuffer expected;
                LayerCache reference;
                for (uint64_t first : {uint64_t{0}, uint64_t{2}, capacity - context - rows, capacity - 1})
                {
                    LayerCache cache;
                    cache.capacity_tokens = capacity;
                    cache.first_slot = first;
                    cache.start_position = 7;
                    cache.token_count = context;
                    cache.columns = 2;
                    cache.keys.resize(capacity * cache.columns);
                    cache.values.resize(capacity * cache.columns);
                    for (uint64_t token = 0; token < context; ++token)
                    {
                        const size_t offset = (first + token) % capacity * cache.columns;
                        cache.keys[offset] = 0.125f * static_cast<float>(token + 1);
                        cache.keys[offset + 1] = -0.2f * static_cast<float>(token + 2);
                        cache.values[offset] = 0.25f * static_cast<float>(token + 1);
                        cache.values[offset + 1] = -0.3f * static_cast<float>(token + 2);
                    }
                    AttentionScratch scratch;
                    ActivationBuffer actual;
                    check(static_cast<bool>(forward_attention(weights, operators, plan, ExecutionBackend::Cpu, 7 + context,
                                                              cache, scratch, input, actual, g_test_optimization_flags)));
                    check(scratch.key_cache.empty());
                    check(scratch.value_cache.empty());
                    if (first == 0)
                    {
                        expected = actual;
                        reference = cache;
                    }
                    check(cache.start_position == reference.start_position);
                    check(cache.token_count == reference.token_count);
                    for (size_t index = 0; index < actual.values().size(); ++index)
                        check_near(actual.values()[index], expected.values()[index], 1e-6f);
                    for (uint64_t token = 0; token < cache.token_count; ++token)
                    {
                        const size_t actual_offset = (cache.first_slot + token) % capacity * cache.columns;
                        const size_t expected_offset = (reference.first_slot + token) % capacity * cache.columns;
                        for (uint32_t column = 0; column < cache.columns; ++column)
                        {
                            check_near(cache.keys[actual_offset + column], reference.keys[expected_offset + column], 1e-6f);
                            check_near(cache.values[actual_offset + column], reference.values[expected_offset + column], 1e-6f);
                        }
                    }
                }
            }

    // Reuse scratch across flash/split-KV calls, PV tails and physical ring boundaries.
    constexpr uint64_t ring_capacity = 1031;
    constexpr uint64_t context = 520;
    constexpr uint32_t width = 80;
    std::vector<float> identity(width * width, 0.0f);
    for (uint32_t i = 0; i < width; ++i)
        identity[i * width + i] = 1.0f;
    plan.head_dimension = width;
    plan.value_head_dimension = width;
    plan.rope_head_dimension = width;
    plan.query_weight = add_float_tensor(weights, "ring_query", {width, width}, identity);
    plan.key_weight = add_float_tensor(weights, "ring_key", {width, width}, identity);
    plan.value_weight = add_float_tensor(weights, "ring_value", {width, width}, identity);
    plan.output_weight = add_float_tensor(weights, "ring_output", {width, width}, identity);
    plan.sinks = add_float_tensor(weights, "ring_sink", {1}, {0.125f});
    CpuOpenMpThreadLimitScope thread_limit;
    AttentionScratch scratch;
    for (uint32_t threads : {8u, 1u, 3u, 8u})
        for (DType dtype : {DType::Float32, DType::BFloat16})
            for (uint32_t window : {0u, 17u})
                for (size_t rows : {8u, 1u, 3u, 1u, 5u})
                {
                    thread_limit.set(threads);
                    plan.kv_cache_dtype = dtype;
                    plan.sliding_window = window;
                    ActivationBuffer input(rows, width);
                    for (size_t row = 0; row < rows; ++row)
                    {
                        input.row(row)[0] = 0.03125f * static_cast<float>(row + 1);
                        input.row(row)[1] = -0.0625f;
                    }
                    ActivationBuffer expected;
                    ActivationBuffer contiguous;
                    for (uint64_t first : {uint64_t{0}, uint64_t{37}, ring_capacity - 3, ring_capacity - 1})
                    {
                        LayerCache cache;
                        cache.dtype = dtype;
                        cache.capacity_tokens = ring_capacity;
                        cache.first_slot = first;
                        cache.start_position = 11;
                        cache.token_count = context;
                        cache.columns = width;
                        if (dtype == DType::Float32)
                        {
                            cache.keys.resize(ring_capacity * width);
                            cache.values.resize(ring_capacity * width);
                        }
                        else
                        {
                            cache.bfloat16_keys.resize(ring_capacity * width);
                            cache.bfloat16_values.resize(ring_capacity * width);
                        }
                        for (uint64_t token = 0; token < context; ++token)
                            for (uint32_t column = 0; column < width; ++column)
                            {
                                const size_t offset = (first + token) % ring_capacity * width + column;
                                const float key = static_cast<float>(static_cast<int>((token * 7 + column) % 31) - 15) * 0.03125f;
                                const float value = static_cast<float>(static_cast<int>((token * 3 + column) % 29) - 14) * 0.0625f;
                                if (dtype == DType::Float32)
                                {
                                    cache.keys[offset] = key;
                                    cache.values[offset] = value;
                                }
                                else
                                {
                                    cache.bfloat16_keys[offset] = float_to_bfloat16(key);
                                    cache.bfloat16_values[offset] = float_to_bfloat16(value);
                                }
                            }
                        if (first == 0)
                        {
                            LayerCache reference_cache = cache;
                            AttentionScratch reference_scratch;
                            check(static_cast<bool>(forward_attention(weights, operators, plan, ExecutionBackend::Cpu, 11 + context,
                                                                      reference_cache, reference_scratch, input, expected,
                                                                      g_test_optimization_flags & ~OptimizationCpuFlashAttention & ~OptimizationCpuSplitKvAttention)));
                        }
                        const float poison = std::numeric_limits<float>::quiet_NaN();
                        std::fill(scratch.workspace.begin(), scratch.workspace.end(), poison);
                        std::fill(scratch.flash_partial_max.begin(), scratch.flash_partial_max.end(), poison);
                        std::fill(scratch.flash_partial_sum.begin(), scratch.flash_partial_sum.end(), poison);
                        std::fill(scratch.flash_partial_output.begin(), scratch.flash_partial_output.end(), poison);
                        ActivationBuffer actual;
                        check(static_cast<bool>(forward_attention(weights, operators, plan, ExecutionBackend::Cpu, 11 + context,
                                                                  cache, scratch, input, actual, g_test_optimization_flags)));
                        if (dtype == DType::BFloat16 && rows > 1
                            && has_flag(g_test_optimization_flags, OptimizationCpuBf16DirectAttention)
                            && has_flag(g_test_optimization_flags, OptimizationCpuFlashAttention))
                        {
                            check(scratch.key_cache.empty());
                            check(scratch.value_cache.size()
                                  == static_cast<size_t>(context + rows) * cache.columns);
                        }
                        if (first == 0)
                            contiguous = actual;
                        check(std::equal(actual.values().begin(), actual.values().end(), contiguous.values().begin()));
                        for (size_t index = 0; index < actual.values().size(); ++index)
                            check_near(actual.values()[index], expected.values()[index], 1e-4f);
                    }
                }
    // Compare grouped QK against independent per-key FP32 attention, including GQA and ring tails.
    thread_limit.set(1);
    for (uint32_t dimension : {32u, 80u, 128u})
    {
        const uint64_t capacity = dimension == 80 ? 521u : 13u;
        const uint32_t width = 4 * dimension;
        const uint32_t cache_width = 2 * dimension;
        std::vector<float> identity(static_cast<size_t>(width) * width, 0.0f);
        for (uint32_t column = 0; column < width; ++column)
            identity[static_cast<size_t>(column) * width + column] = 1.0f;
        const std::string prefix = "qk_batch_" + std::to_string(dimension);
        AttentionBlockPlan batch_plan;
        batch_plan.kind = AttentionKind::Standard;
        batch_plan.head_count = 4;
        batch_plan.kv_head_count = 2;
        batch_plan.head_dimension = dimension;
        batch_plan.value_head_dimension = dimension;
        batch_plan.kv_cache_dtype = DType::BFloat16;
        batch_plan.query_weight = add_float_tensor(weights, prefix + "query", {width, width}, identity);
        batch_plan.output_weight = add_float_tensor(weights, prefix + "output", {width, width}, identity);
        identity.resize(static_cast<size_t>(cache_width) * width);
        batch_plan.key_weight = add_float_tensor(weights, prefix + "key", {cache_width, width}, identity);
        batch_plan.value_weight = add_float_tensor(weights, prefix + "value", {cache_width, width}, identity);
        if (dimension == 80)
            batch_plan.sinks = add_float_tensor(weights, prefix + "sink", {4}, {0.125f, -0.0625f, 0.1875f, 0.03125f});
        const uint64_t flags = g_test_optimization_flags
                               | OptimizationCpuBf16DirectAttention
                               | OptimizationCpuFlashAttention
                               | OptimizationCpuSplitKvAttention;
        const uint64_t reference_flags = flags & ~OptimizationCpuBf16DirectAttention
                                         & ~OptimizationCpuFlashAttention
                                         & ~OptimizationCpuSplitKvAttention;
        const std::vector<uint64_t> contexts = dimension == 80
                                                   ? std::vector<uint64_t>{63, 64, 65, 511, 512}
                                                   : std::vector<uint64_t>{3, 4, 6, 7, 8};
        const std::vector<uint64_t> first_slots = dimension == 80
                                                      ? std::vector<uint64_t>{0, capacity - 59, capacity - 1}
                                                      : std::vector<uint64_t>{0, capacity - 3, capacity - 1};
        const std::vector<uint32_t> thread_counts = dimension == 80
                                                        ? std::vector<uint32_t>{1, 8}
                                                        : std::vector<uint32_t>{1};
        for (uint32_t threads : thread_counts)
        {
            thread_limit.set(threads);
            AttentionScratch batch_scratch;
            AttentionScratch fallback_scratch;
            for (uint64_t context : contexts)
                for (uint64_t first : first_slots)
                {
                    const uint64_t stored_context = dimension == 80 ? context - 1 : context;
                    LayerCache cache;
                    cache.dtype = DType::BFloat16;
                    cache.columns = cache_width;
                    cache.capacity_tokens = capacity;
                    cache.first_slot = first;
                    cache.start_position = 7;
                    cache.token_count = stored_context;
                    cache.bfloat16_keys.resize(capacity * cache_width);
                    cache.bfloat16_values.resize(capacity * cache_width);
                    for (uint64_t token = 0; token < stored_context; ++token)
                        for (uint32_t column = 0; column < cache_width; ++column)
                        {
                            const size_t offset = (first + token) % capacity * cache_width + column;
                            cache.bfloat16_keys[offset] = float_to_bfloat16(static_cast<float>(static_cast<int>((token * 7 + column * 3) % 29) - 14) * 0.03125f);
                            cache.bfloat16_values[offset] = float_to_bfloat16(static_cast<float>(static_cast<int>((token * 11 + column * 5) % 31) - 15) * 0.0625f);
                        }
                    LayerCache reference_cache = cache;
                    ActivationBuffer input(1, width);
                    for (uint32_t column = 0; column < width; ++column)
                        input.row(0)[column] = static_cast<float>(static_cast<int>((column * 13) % 37) - 18) * 0.0234375f;
                    AttentionScratch& active_scratch = dimension == 80 && context == 512
                                                           ? fallback_scratch
                                                           : batch_scratch;
                    const float poison = std::numeric_limits<float>::quiet_NaN();
                    std::fill(active_scratch.workspace.begin(), active_scratch.workspace.end(), poison);
                    std::fill(active_scratch.logits.begin(), active_scratch.logits.end(), poison);
                    std::fill(active_scratch.flash_partial_max.begin(), active_scratch.flash_partial_max.end(), poison);
                    std::fill(active_scratch.flash_partial_sum.begin(), active_scratch.flash_partial_sum.end(), poison);
                    std::fill(active_scratch.flash_partial_output.begin(), active_scratch.flash_partial_output.end(), poison);
                    AttentionScratch reference_scratch;
                    ActivationBuffer expected;
                    ActivationBuffer actual;
                    check(static_cast<bool>(forward_attention(weights, operators, batch_plan, ExecutionBackend::Cpu, 7 + stored_context,
                                                              reference_cache, reference_scratch, input, expected,
                                                              reference_flags)));
                    check(static_cast<bool>(forward_attention(weights, operators, batch_plan, ExecutionBackend::Cpu, 7 + stored_context,
                                                              cache, active_scratch, input, actual, flags)));
                    check(active_scratch.key_cache.empty());
                    check(active_scratch.value_cache.empty());
                    if (dimension == 80)
                    {
                        if (context >= 64 && context < 512)
                            check(!active_scratch.workspace.empty());
                        else if (context == 63 || context == 512)
                            check(active_scratch.workspace.empty());
                    }
                    check(cache.first_slot == reference_cache.first_slot);
                    check(cache.token_count == reference_cache.token_count);
                    check(cache.bfloat16_keys == reference_cache.bfloat16_keys);
                    check(cache.bfloat16_values == reference_cache.bfloat16_values);
                    check(actual.rows() == expected.rows());
                    check(actual.columns() == expected.columns());
                    for (size_t index = 0; index < actual.values().size(); ++index)
                        check_near(actual.values()[index], expected.values()[index], 1e-4f);
                }
        }
    }
}

void test_qsa_prefill_decode_continuation()
{
    WeightStore weights;
    CompiledOperatorTable operators;
    AttentionBlockPlan plan;
    plan.norm_epsilon = 1e-6f;
    plan.kv_cache_dtype = DType::BFloat16;
    plan.kind = AttentionKind::Standard;
    plan.head_count = 1;
    plan.kv_head_count = 1;
    plan.head_dimension = 2;
    plan.value_head_dimension = 2;
    plan.rope_head_dimension = 2;
    plan.rope_theta = 10000.0f;
    plan.rope_scaling_factor = 1.0f;
    plan.norm_weight_offset = 1.0f;
    plan.index_head_count = 1;
    plan.index_head_dimension = 2;
    plan.index_top_k = 1;
    plan.compression_ratio = 2;
    plan.flags = AttentionBlockQsa | AttentionBlockExternalResidual;
    const std::vector<float> identity = {
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    plan.query_weight = add_bfloat16_tensor(weights, "qsa_query", {2, 2}, identity);
    plan.key_weight = add_bfloat16_tensor(weights, "qsa_key", {2, 2}, identity);
    plan.value_weight = add_bfloat16_tensor(weights, "qsa_value", {2, 2}, identity);
    plan.output_weight = add_bfloat16_tensor(weights, "qsa_output", {2, 2}, identity);
    plan.qsa_query_key_weight = add_bfloat16_tensor(weights,
                                                    "qsa_query_key",
                                                    {4, 2},
                                                    {1.0f, 0.0f,
                                                     0.0f, 1.0f,
                                                     1.0f, 0.0f,
                                                     0.0f, 1.0f});
    plan.qsa_query_norm_weight = add_bfloat16_tensor(weights, "qsa_query_norm", {2}, {0.0f, 0.0f});
    plan.qsa_key_norm_weight = add_bfloat16_tensor(weights, "qsa_key_norm", {2}, {0.0f, 0.0f});

    ActivationBuffer input(6, 2);
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

    const auto reference_qsa_indices = [&weights](const AttentionBlockPlan& reference_plan,
                                                  const ActivationBuffer& raw_input,
                                                  const ActivationBuffer& query,
                                                  uint64_t first_query_position,
                                                  uint64_t cache_start_position,
                                                  uint64_t available_tokens) {
        std::vector<std::vector<uint32_t>> expected(query.rows());
        const uint32_t dimension = reference_plan.index_head_dimension;
        const uint32_t ratio = reference_plan.compression_ratio;
        const float scale = 1.0f / std::sqrt(static_cast<float>(dimension));
        const TensorData& key_norm = weights.at(reference_plan.qsa_key_norm_weight);
        for (size_t query_index = 0; query_index < query.rows(); ++query_index)
        {
            const uint64_t query_position = first_query_position + query_index;
            const uint64_t visible_count = query_position < cache_start_position
                                               ? 0
                                               : std::min<uint64_t>(available_tokens,
                                                                    query_position - cache_start_position + 1);
            const uint64_t complete_blocks = visible_count / ratio;
            std::vector<std::pair<float, uint32_t>> scores;
            scores.reserve(static_cast<size_t>(complete_blocks));
            for (uint32_t block = 0; block < complete_blocks; ++block)
            {
                std::array<float, 2> pooled = {};
                const uint64_t first_token = static_cast<uint64_t>(block) * ratio;
                for (uint32_t token = 0; token < ratio; ++token)
                {
                    for (uint32_t column = 0; column < dimension; ++column)
                    {
                        pooled[column] += bfloat16_to_float(float_to_bfloat16(raw_input.row(first_token + token)[column]));
                    }
                }
                const float inverse_ratio = 1.0f / static_cast<float>(ratio);
                for (float& value : pooled)
                    value = bfloat16_to_float(float_to_bfloat16(value * inverse_ratio));
                float square_sum = 0.0f;
                for (float value : pooled)
                    square_sum += value * value;
                const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(dimension) + reference_plan.norm_epsilon);
                for (uint32_t column = 0; column < dimension; ++column)
                {
                    pooled[column] *= inverse_rms
                                      * (bfloat16_to_float(key_norm.bfloat16_values()[column])
                                         + reference_plan.norm_weight_offset);
                }
                // The test fixture uses a two-element index key, whose RoPE
                // frequency is one; retain the old block-start position rule.
                const float angle = static_cast<float>(cache_start_position + first_token);
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const float first = pooled[0];
                const float second = pooled[1];
                pooled[0] = first * cosine - second * sine;
                pooled[1] = second * cosine + first * sine;
                float score = std::max(0.0f,
                                       float_dot(query.row(query_index), pooled.data(), dimension));
                scores.emplace_back(score * scale, block);
            }
            const size_t selected_count = std::min<size_t>(reference_plan.index_top_k, scores.size());
            std::partial_sort(scores.begin(), scores.begin() + selected_count, scores.end(),
                              [](const auto& left, const auto& right) {
                                  if (left.first != right.first)
                                      return left.first > right.first;
                                  return left.second < right.second;
                              });
            for (size_t index = 0; index < selected_count; ++index)
            {
                const uint64_t first_token = static_cast<uint64_t>(scores[index].second) * ratio;
                for (uint32_t token = 0; token < ratio; ++token)
                    expected[query_index].push_back(static_cast<uint32_t>(first_token + token));
            }
            for (uint64_t token = complete_blocks * ratio; token < visible_count; ++token)
                expected[query_index].push_back(static_cast<uint32_t>(token));
            std::sort(expected[query_index].begin(), expected[query_index].end());
        }
        return expected;
    };

    const auto check_qsa_selection = [](const AttentionScratch& scratch,
                                        const std::vector<std::vector<uint32_t>>& expected) {
        check(scratch.qsa_selected_offsets.size() == expected.size() + 1);
        for (size_t query_index = 0; query_index < expected.size(); ++query_index)
        {
            const auto begin = scratch.qsa_selected_indices.begin()
                               + scratch.qsa_selected_offsets[query_index];
            const auto end = scratch.qsa_selected_indices.begin()
                             + scratch.qsa_selected_offsets[query_index + 1];
            check(std::vector<uint32_t>(begin, end) == expected[query_index]);
        }
    };

    LayerCache prefill_cache;
    AttentionScratch prefill_scratch;
    ActivationBuffer prefill_output;
    constexpr uint64_t prefill_position = 11;
    check(static_cast<bool>(forward_attention(weights,
                                              operators,
                                              plan,
                                              ExecutionBackend::Cpu,
                                              prefill_position,
                                              prefill_cache,
                                              prefill_scratch,
                                              input,
                                              prefill_output,
                                              0)));
    check_qsa_selection(prefill_scratch,
                        reference_qsa_indices(plan, input, prefill_scratch.qsa_query,
                                              prefill_position, prefill_position, input.rows()));

    LayerCache decode_cache;
    AttentionScratch decode_scratch;
    ActivationBuffer decode_output(input.rows(), 2);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        ActivationBuffer token(1, 2);
        std::copy_n(input.row(row), 2, token.row(0));
        ActivationBuffer token_output;
        check(static_cast<bool>(forward_attention(weights,
                                                  operators,
                                                  plan,
                                                  ExecutionBackend::Cpu,
                                                  prefill_position + row,
                                                  decode_cache,
                                                  decode_scratch,
                                                  token,
                                                  token_output,
                                                  0)));
        check_qsa_selection(decode_scratch,
                            reference_qsa_indices(plan, input, decode_scratch.qsa_query,
                                                  prefill_position + row,
                                                  prefill_position, row + 1));
        std::copy_n(token_output.row(0), 2, decode_output.row(row));
    }
    check(prefill_cache.token_count == input.rows());
    check(prefill_cache.qsa_block_keys.size()
          == (input.rows() / plan.compression_ratio) * plan.index_head_dimension);
    check(prefill_cache.qsa_index_key_tail.size()
          == (input.rows() % plan.compression_ratio) * plan.index_head_dimension);
    check(prefill_cache.qsa_block_keys == decode_cache.qsa_block_keys);
    check(prefill_cache.qsa_index_key_tail == decode_cache.qsa_index_key_tail);
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

    AttentionBlockPlan all_blocks_plan = plan;
    all_blocks_plan.index_top_k = static_cast<uint32_t>(input.rows());
    LayerCache all_blocks_cache;
    AttentionScratch all_blocks_scratch;
    ActivationBuffer all_blocks_output;
    check(static_cast<bool>(forward_attention(weights,
                                              operators,
                                              all_blocks_plan,
                                              ExecutionBackend::Cpu,
                                              prefill_position,
                                              all_blocks_cache,
                                              all_blocks_scratch,
                                              input,
                                              all_blocks_output,
                                              0)));
    AttentionBlockPlan unfiltered_plan = all_blocks_plan;
    unfiltered_plan.flags &= ~static_cast<uint32_t>(AttentionBlockQsa);
    LayerCache unfiltered_cache;
    AttentionScratch unfiltered_scratch;
    ActivationBuffer unfiltered_output;
    check(static_cast<bool>(forward_attention(weights,
                                              operators,
                                              unfiltered_plan,
                                              ExecutionBackend::Cpu,
                                              prefill_position,
                                              unfiltered_cache,
                                              unfiltered_scratch,
                                              input,
                                              unfiltered_output,
                                              0)));
    for (size_t index = 0; index < all_blocks_output.values().size(); ++index)
        check_near(all_blocks_output.values()[index], unfiltered_output.values()[index], 1e-6f);

    AttentionBlockPlan boundary_plan = plan;
    boundary_plan.compression_ratio = 4;
    ActivationBuffer boundary_input(5, 2);
    for (size_t row = 0; row < boundary_input.rows(); ++row)
        std::copy_n(input.row(row), 2, boundary_input.row(row));
    constexpr uint64_t boundary_position = 23;
    LayerCache boundary_prefill_cache;
    AttentionScratch boundary_prefill_scratch;
    ActivationBuffer boundary_prefill_output;
    check(static_cast<bool>(forward_attention(weights,
                                              operators,
                                              boundary_plan,
                                              ExecutionBackend::Cpu,
                                              boundary_position,
                                              boundary_prefill_cache,
                                              boundary_prefill_scratch,
                                              boundary_input,
                                              boundary_prefill_output,
                                              0)));
    check_qsa_selection(boundary_prefill_scratch,
                        reference_qsa_indices(boundary_plan, boundary_input,
                                              boundary_prefill_scratch.qsa_query,
                                              boundary_position, boundary_position,
                                              boundary_input.rows()));

    LayerCache qsa_accounted = boundary_prefill_cache;
    LayerCache qsa_cleared = boundary_prefill_cache;
    const uint64_t qsa_allocated = static_cast<uint64_t>(qsa_accounted.qsa_block_keys.capacity()) * sizeof(float)
                                   + static_cast<uint64_t>(qsa_accounted.qsa_index_key_tail.capacity()) * sizeof(uint16_t);
    const uint64_t qsa_logical = static_cast<uint64_t>(qsa_accounted.qsa_block_keys.size()) * sizeof(float)
                                 + static_cast<uint64_t>(qsa_accounted.qsa_index_key_tail.size()) * sizeof(uint16_t);
    std::vector<float>().swap(qsa_cleared.qsa_block_keys);
    std::vector<uint16_t>().swap(qsa_cleared.qsa_index_key_tail);
    check(qsa_accounted.allocated_bytes() - qsa_cleared.allocated_bytes()
          == qsa_allocated);
    check(qsa_accounted.logical_bytes() - qsa_cleared.logical_bytes()
          == qsa_logical);

    LayerCache segmented_cache;
    AttentionScratch context_scratch;
    ActivationBuffer context_prefix(3, 2);
    for (size_t row = 0; row < context_prefix.rows(); ++row)
        std::copy_n(boundary_input.row(row), 2, context_prefix.row(row));
    check(static_cast<bool>(append_attention_context(weights,
                                                     operators,
                                                     boundary_plan,
                                                     ExecutionBackend::Cpu,
                                                     boundary_position,
                                                     segmented_cache,
                                                     context_scratch,
                                                     context_prefix,
                                                     0)));
    check(segmented_cache.qsa_block_keys.empty());
    check(segmented_cache.qsa_index_key_tail.size() == 3 * boundary_plan.index_head_dimension);
    ActivationBuffer segmented_output(5, 2);
    for (size_t row = 3; row < boundary_input.rows(); ++row)
    {
        ActivationBuffer token(1, 2);
        std::copy_n(boundary_input.row(row), 2, token.row(0));
        ActivationBuffer token_output;
        check(static_cast<bool>(forward_attention(weights,
                                                  operators,
                                                  boundary_plan,
                                                  ExecutionBackend::Cpu,
                                                  boundary_position + row,
                                                  segmented_cache,
                                                  context_scratch,
                                                  token,
                                                  token_output,
                                                  0)));
        check_qsa_selection(context_scratch,
                            reference_qsa_indices(boundary_plan, boundary_input,
                                                  context_scratch.qsa_query,
                                                  boundary_position + row,
                                                  boundary_position, row + 1));
        std::copy_n(token_output.row(0), 2, segmented_output.row(row));
        if (row == 3)
        {
            check(segmented_cache.qsa_block_keys.size()
                  == boundary_plan.index_head_dimension);
            check(segmented_cache.qsa_index_key_tail.empty());
        }
        else
        {
            check(segmented_cache.qsa_block_keys.size()
                  == boundary_plan.index_head_dimension);
            check(segmented_cache.qsa_index_key_tail.size()
                  == boundary_plan.index_head_dimension);
        }
        for (uint32_t column = 0; column < boundary_prefill_output.columns(); ++column)
            check_near(token_output.row(0)[column], boundary_prefill_output.row(row)[column], 1e-6f);
    }

    segmented_cache = LayerCache{};
    AttentionBlockPlan single_block_plan = plan;
    single_block_plan.compression_ratio = 1;
    constexpr uint64_t reset_position = 41;
    ActivationBuffer reset_input(1, 2);
    std::copy_n(input.row(0), 2, reset_input.row(0));
    ActivationBuffer reset_output;
    check(static_cast<bool>(forward_attention(weights,
                                              operators,
                                              single_block_plan,
                                              ExecutionBackend::Cpu,
                                              reset_position,
                                              segmented_cache,
                                              context_scratch,
                                              reset_input,
                                              reset_output,
                                              0)));
    check(segmented_cache.start_position == reset_position);
    check(segmented_cache.qsa_block_keys.size() == single_block_plan.index_head_dimension);
    check(segmented_cache.qsa_index_key_tail.empty());
    check_qsa_selection(context_scratch,
                        reference_qsa_indices(single_block_plan, reset_input,
                                              context_scratch.qsa_query,
                                              reset_position, reset_position, 1));

    LayerCache transaction_cache;
    transaction_cache.columns = plan.kv_head_count * plan.head_dimension;
    transaction_cache.dtype = plan.kv_cache_dtype;
    transaction_cache.transaction.active = true;
    AttentionScratch transaction_scratch;
    ActivationBuffer transaction_output;
    auto transaction_status = forward_attention(weights,
                                                operators,
                                                plan,
                                                ExecutionBackend::Cpu,
                                                prefill_position,
                                                transaction_cache,
                                                transaction_scratch,
                                                reset_input,
                                                transaction_output,
                                                0);
    check(!transaction_status);
    check(transaction_status.error().code == ErrorCode::UnsupportedModel);
    check(transaction_cache.qsa_block_keys.empty());
    check(transaction_cache.qsa_index_key_tail.empty());

    AttentionBlockPlan float_plan = plan;
    float_plan.kv_cache_dtype = DType::Float32;
    ActivationBuffer float_expected;
    std::vector<uint32_t> expected_indices;
    for (uint64_t first : {0u, 2u, 10u, 15u})
    {
        LayerCache cache;
        cache.capacity_tokens = 16;
        cache.first_slot = first;
        cache.columns = 2;
        cache.keys.resize(32);
        cache.values.resize(32);
        AttentionScratch scratch;
        ActivationBuffer actual;
        check(static_cast<bool>(forward_attention(weights, operators, float_plan, ExecutionBackend::Cpu, 0,
                                                  cache, scratch, input, actual, 0)));
        if (first == 0)
        {
            float_expected = actual;
            expected_indices = scratch.qsa_selected_indices;
        }
        check(scratch.qsa_selected_indices == expected_indices);
        check(scratch.key_cache.empty());
        check(scratch.value_cache.empty());
        for (size_t index = 0; index < actual.values().size(); ++index)
            check_near(actual.values()[index], float_expected.values()[index], 1e-6f);
    }

    AttentionBlockPlan long_plan = plan;
    long_plan.index_top_k = 512;
    long_plan.compression_ratio = 4;
    constexpr uint64_t existing_tokens = 262142;
    LayerCache long_cache;
    long_cache.columns = 2;
    long_cache.dtype = DType::BFloat16;
    long_cache.capacity_tokens = existing_tokens;
    long_cache.token_count = existing_tokens;
    long_cache.bfloat16_keys.assign(existing_tokens * 2, 0);
    long_cache.bfloat16_values.assign(existing_tokens * 2, 0);
    long_cache.qsa_block_keys.assign((existing_tokens / long_plan.compression_ratio)
                                         * long_plan.index_head_dimension,
                                     0.0f);
    long_cache.qsa_index_key_tail.assign((existing_tokens % long_plan.compression_ratio)
                                             * long_plan.index_head_dimension,
                                         0);
    AttentionScratch long_scratch;
    ActivationBuffer long_input(2, 2);
    long_input.row(0)[0] = 1.0f;
    long_input.row(1)[1] = 1.0f;
    ActivationBuffer long_output;
    check(static_cast<bool>(forward_attention(weights,
                                              operators,
                                              long_plan,
                                              ExecutionBackend::Cpu,
                                              existing_tokens,
                                              long_cache,
                                              long_scratch,
                                              long_input,
                                              long_output,
                                              OptimizationCpuBf16DirectAttention)));
    check(long_cache.token_count == existing_tokens + long_input.rows());
    check(long_scratch.qsa_selected_offsets.size() == 3);
    const uint32_t maximum_selected_per_query = long_plan.index_top_k * long_plan.compression_ratio
                                                + long_plan.compression_ratio - 1;
    check(long_scratch.qsa_selected_indices.size()
          <= 2 * maximum_selected_per_query);
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
    const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
    WeightStore weights;
    CompiledOperatorTable operators;
    AttentionBlockPlan plan;
    plan.norm_epsilon = 1e-6f;
    plan.kind = AttentionKind::GatedDeltaNet;
    plan.head_count = 1;
    plan.kv_head_count = 1;
    plan.head_dimension = 2;
    plan.value_head_dimension = 2;
    plan.convolution_kernel_size = 2;
    plan.norm_weight_offset = 1.0f;
    plan.pre_attention_norm_weight = add_float_tensor(weights, "pre_norm", {2}, {0.0f, 0.0f});
    plan.delta_qkv_weight = add_float_tensor(weights,
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
    plan.delta_convolution_weight = add_float_tensor(weights,
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

    const AttentionBlockPlan original_plan = plan;
    for (bool normalize : {false, true})
        for (bool external_residual : {false, true})
            for (int alias = 0; alias < 3; ++alias)
            {
                plan.pre_attention_norm_weight = normalize ? original_plan.pre_attention_norm_weight : invalid_tensor_handle;
                plan.flags = external_residual ? AttentionBlockExternalResidual : 0;
                LayerCache reference_cache;
                LayerCache alias_cache;
                GatedDeltaScratch reference_scratch;
                GatedDeltaScratch alias_scratch;
                ActivationBuffer expected;
                for (size_t rows : {3u, 1u, 2u})
                {
                    ActivationBuffer input(rows, 2);
                    for (size_t row = 0; row < rows; ++row)
                    {
                        input.row(row)[0] = 0.15f * static_cast<float>(row + 1);
                        input.row(row)[1] = -0.3f;
                    }
                    ActivationBuffer aliased_input = input;
                    ActivationBuffer& actual = alias == 0   ? aliased_input
                                               : alias == 1 ? alias_scratch.recurrent_output
                                                            : alias_scratch.projected;
                    check(static_cast<bool>(forward_gated_delta(weights, operators, plan, ExecutionBackend::Cpu, reference_cache, reference_scratch, input, expected,
                                                                g_test_optimization_flags)));
                    check(static_cast<bool>(forward_gated_delta(weights, operators, plan, ExecutionBackend::Cpu, alias_cache, alias_scratch, aliased_input, actual,
                                                                g_test_optimization_flags)));
                    check(actual.rows() == expected.rows());
                    check(actual.columns() == expected.columns());
                    for (size_t index = 0; index < expected.values().size(); ++index)
                        check_near(actual.values()[index], expected.values()[index], 1e-6f);
                    check(alias_cache.gated_delta_token_count == reference_cache.gated_delta_token_count);
                    check(alias_cache.gated_delta_convolution == reference_cache.gated_delta_convolution);
                    check(alias_cache.gated_delta_recurrent == reference_cache.gated_delta_recurrent);
                }
            }
    plan = original_plan;

    ActivationBuffer input(2, 2);
    input.row(0)[0] = 0.75f;
    input.row(0)[1] = -0.25f;
    input.row(1)[0] = -0.40f;
    input.row(1)[1] = 0.90f;

    LayerCache prefill_cache;
    GatedDeltaScratch prefill_scratch;
    ActivationBuffer prefill_output;
    check(static_cast<bool>(forward_gated_delta(weights,
                                                operators,
                                                plan,
                                                ExecutionBackend::Cpu,
                                                prefill_cache,
                                                prefill_scratch,
                                                input,
                                                prefill_output,
                                                g_test_optimization_flags)));

    LayerCache decode_cache;
    GatedDeltaScratch decode_scratch;
    ActivationBuffer decode_output(2, 2);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        ActivationBuffer token(1, 2);
        std::copy_n(input.row(row), 2, token.row(0));
        ActivationBuffer token_output;
        check(static_cast<bool>(forward_gated_delta(weights,
                                                    operators,
                                                    plan,
                                                    ExecutionBackend::Cpu,
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

    const CompiledOperatorHandle fused_delta_handle = operators.allocate();
#if NCNN_MOE_WITH_NCNN && NCNN_MOE_WITH_VULKAN
    if (get_gpu_count() > 0)
    {
        auto fused_operator = Linear::create_fused({
                                                       &weights.at(plan.delta_qkv_weight),
                                                       &weights.at(plan.delta_z_weight),
                                                       &weights.at(plan.delta_beta_weight),
                                                       &weights.at(plan.delta_alpha_weight),
                                                   },
                                                   {nullptr, nullptr, nullptr, nullptr}, get_default_gpu_index(), vulkan_runtime, g_test_optimization_flags);
        check(static_cast<bool>(fused_operator));
        operators.at_mutable(fused_delta_handle).linear = std::move(fused_operator);
    }
#endif
    plan.fused_delta_input_operator = fused_delta_handle;
    LayerCache fused_cache;
    GatedDeltaScratch fused_scratch;
    ActivationBuffer fused_output;
    check(static_cast<bool>(forward_gated_delta(weights,
                                                operators,
                                                plan,
                                                ExecutionBackend::Cpu,
                                                fused_cache,
                                                fused_scratch,
                                                input,
                                                fused_output,
                                                g_test_optimization_flags)));
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (size_t column = 0; column < input.columns(); ++column)
        {
            check_near(fused_output.row(row)[column],
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
        check_near(fused_cache.gated_delta_convolution[index],
                   prefill_cache.gated_delta_convolution[index],
                   1e-5f);
    }
    check(fused_cache.gated_delta_recurrent.size()
          == prefill_cache.gated_delta_recurrent.size());
    for (size_t index = 0;
         index < fused_cache.gated_delta_recurrent.size();
         ++index)
    {
        check_near(fused_cache.gated_delta_recurrent[index],
                   prefill_cache.gated_delta_recurrent[index],
                   1e-5f);
    }

    LayerCache first_row_cache;
    GatedDeltaScratch first_row_scratch;
    ActivationBuffer first_row_input(1, 2);
    std::copy_n(input.row(0), 2, first_row_input.row(0));
    ActivationBuffer first_row_output;
    check(static_cast<bool>(forward_gated_delta(weights,
                                                operators,
                                                plan,
                                                ExecutionBackend::Cpu,
                                                first_row_cache,
                                                first_row_scratch,
                                                first_row_input,
                                                first_row_output,
                                                g_test_optimization_flags)));

    std::array<LayerCache, 1> committed_cache;
    check(static_cast<bool>(begin_state_cache_transaction(committed_cache, 2)));
    auto duplicate_transaction = begin_state_cache_transaction(committed_cache,
                                                               2);
    check(static_cast<bool>(!duplicate_transaction));
    check(static_cast<bool>(committed_cache.front().transaction.active));
    GatedDeltaScratch committed_scratch;
    ActivationBuffer committed_output;
    check(static_cast<bool>(forward_gated_delta(weights,
                                                operators,
                                                plan,
                                                ExecutionBackend::Cpu,
                                                committed_cache.front(),
                                                committed_scratch,
                                                input,
                                                committed_output,
                                                g_test_optimization_flags)));
    auto committed = finish_state_cache_transaction(committed_cache,
                                                    1);
    check(static_cast<bool>(committed));
    check(committed_cache.front().gated_delta_token_count == 1);
    check(committed_cache.front().gated_delta_convolution
          == first_row_cache.gated_delta_convolution);
    check(committed_cache.front().gated_delta_recurrent
          == first_row_cache.gated_delta_recurrent);
    check(static_cast<bool>(!committed_cache.front().transaction.active));
    check(static_cast<bool>(committed_cache.front().transaction.rows.empty()));
    check(static_cast<bool>(committed_cache.front().transaction.initial.gated_delta_recurrent.empty()));

    std::array<LayerCache, 1> rolled_back_cache;
    check(static_cast<bool>(begin_state_cache_transaction(rolled_back_cache, 2)));
    GatedDeltaScratch rolled_back_scratch;
    ActivationBuffer rolled_back_output;
    check(static_cast<bool>(forward_gated_delta(weights,
                                                operators,
                                                plan,
                                                ExecutionBackend::Cpu,
                                                rolled_back_cache.front(),
                                                rolled_back_scratch,
                                                input,
                                                rolled_back_output,
                                                g_test_optimization_flags)));
    auto rolled_back = finish_state_cache_transaction(rolled_back_cache,
                                                      0);
    check(static_cast<bool>(rolled_back));
    check(rolled_back_cache.front().gated_delta_token_count == 0);
    check(rolled_back_cache.front().gated_delta_convolution.empty());
    check(rolled_back_cache.front().gated_delta_recurrent.empty());
    check(static_cast<bool>(!rolled_back_cache.front().transaction.active));
    check(static_cast<bool>(rolled_back_cache.front().transaction.rows.empty()));

    std::array<LayerCache, 1> standard_cache;
    standard_cache.front().token_count = 5;
    check(static_cast<bool>(begin_state_cache_transaction(standard_cache, 4)));
    standard_cache.front().token_count = 9;
    record_standard_cache_transaction_rows(standard_cache.front(),
                                           4);
    auto standard_committed = finish_state_cache_transaction(standard_cache,
                                                             2);
    check(static_cast<bool>(standard_committed));
    check(standard_cache.front().token_count == 7);

    std::array<LayerCache, 1> sliding_cache;
    sliding_cache.front().token_count = 4;
    sliding_cache.front().capacity_tokens = 4;
    sliding_cache.front().first_slot = 1;
    auto sliding_transaction = begin_state_cache_transaction(sliding_cache,
                                                             2);
    check(static_cast<bool>(!sliding_transaction));
    check(static_cast<bool>(sliding_transaction.error().code == ErrorCode::UnsupportedModel));
    check(static_cast<bool>(!sliding_cache.front().transaction.active));

    LayerCache poisoned_attention_cache;
    poisoned_attention_cache.vulkan_attention_state_unknown = true;
    AttentionScratch poisoned_attention_scratch;
    ActivationBuffer poisoned_attention_hidden(1, 1);
    ActivationBuffer poisoned_attention_output;
    WeightStore poisoned_attention_weights;
    CompiledOperatorTable poisoned_attention_operators;
    AttentionBlockPlan poisoned_attention_plan;
    poisoned_attention_plan.kind = AttentionKind::Standard;
    poisoned_attention_plan.norm_epsilon = 1e-6f;
    auto poisoned_attention = forward_attention(poisoned_attention_weights,
                                                poisoned_attention_operators,
                                                poisoned_attention_plan,
                                                ExecutionBackend::Cpu,
                                                0,
                                                poisoned_attention_cache,
                                                poisoned_attention_scratch,
                                                poisoned_attention_hidden,
                                                poisoned_attention_output,
                                                g_test_optimization_flags);
    check(static_cast<bool>(!poisoned_attention));
    check(static_cast<bool>(poisoned_attention.error().code == ErrorCode::InternalError));

    CompiledModel cpu_gdn_release_model;
    const std::array<TensorHandle, 4> cpu_gdn_input_weights = {
        add_float_tensor(cpu_gdn_release_model.weights, "gdn_qkv", {1, 1}, {1.0f}),
        add_float_tensor(cpu_gdn_release_model.weights, "gdn_z", {1, 1}, {1.0f}),
        add_float_tensor(cpu_gdn_release_model.weights, "gdn_beta", {1, 1}, {1.0f}),
        add_float_tensor(cpu_gdn_release_model.weights, "gdn_alpha", {1, 1}, {1.0f}),
    };
    cpu_gdn_release_model.operators.bind_weight_count(cpu_gdn_release_model.weights.size());
    const CompiledOperatorHandle cpu_gdn_handle = cpu_gdn_release_model.operators.allocate();
    cpu_gdn_release_model.operators.at_mutable(cpu_gdn_handle).linear = operators.at(fused_delta_handle).linear;
    CompiledLayerPlan cpu_gdn_layer;
    cpu_gdn_layer.attention.kind = AttentionKind::GatedDeltaNet;
    cpu_gdn_layer.attention.fused_delta_input_operator = cpu_gdn_handle;
    cpu_gdn_layer.attention.delta_qkv_weight = cpu_gdn_input_weights[0];
    cpu_gdn_layer.attention.delta_z_weight = cpu_gdn_input_weights[1];
    cpu_gdn_layer.attention.delta_beta_weight = cpu_gdn_input_weights[2];
    cpu_gdn_layer.attention.delta_alpha_weight = cpu_gdn_input_weights[3];
    cpu_gdn_release_model.net.layer_plans.push_back(std::move(cpu_gdn_layer));
    release_host_copies(cpu_gdn_release_model);
    for (TensorHandle handle : cpu_gdn_input_weights)
        check(!cpu_gdn_release_model.weights.at(handle).float32_data.empty());

    // Grouped heads must preserve token order when a whole prefill is assigned
    // to each worker. Compare against one-token execution and partial rollback.
    CpuOpenMpThreadLimitScope thread_limit;
    for (DType dtype : {DType::Float32, DType::BFloat16})
    {
        WeightStore grouped_weights;
        AttentionBlockPlan grouped;
        grouped.kind = AttentionKind::GatedDeltaNet;
        grouped.norm_epsilon = 1e-6f;
        grouped.head_count = 6;
        grouped.kv_head_count = 3;
        grouped.head_dimension = 32;
        grouped.value_head_dimension = 32;
        grouped.convolution_kernel_size = 4;
        constexpr uint32_t width = 8;
        constexpr uint32_t qkv_width = 384;
        constexpr uint32_t value_width = 192;
        const auto add_weight = [&](const std::string& name, const std::vector<uint32_t>& shape, size_t count, int seed) {
            std::vector<float> values(count);
            for (size_t i = 0; i < count; ++i)
                values[i] = static_cast<float>(static_cast<int>((i * 7 + seed) % 23) - 11) / 128.0f;
            return dtype == DType::Float32 ? add_float_tensor(grouped_weights, name, shape, values)
                                           : add_bfloat16_tensor(grouped_weights, name, shape, values);
        };
        grouped.delta_qkv_weight = add_weight("qkv", {qkv_width, width}, qkv_width * width, 1);
        grouped.delta_z_weight = add_weight("z", {value_width, width}, value_width * width, 3);
        grouped.delta_beta_weight = add_weight("beta", {6, width}, 6 * width, 5);
        grouped.delta_alpha_weight = add_weight("alpha", {6, width}, 6 * width, 7);
        grouped.delta_convolution_weight = add_weight("conv", {qkv_width, 1, 4}, qkv_width * 4, 9);
        grouped.delta_time_bias = add_weight("time", {6}, 6, 11);
        grouped.delta_decay_log = add_weight("decay", {6}, 6, 13);
        grouped.delta_norm_weight = dtype == DType::Float32
                                        ? add_float_tensor(grouped_weights, "norm", {32}, std::vector<float>(32, 1.0f))
                                        : add_bfloat16_tensor(grouped_weights, "norm", {32}, std::vector<float>(32, 1.0f));
        grouped.output_weight = add_weight("output", {width, value_width}, width * value_width, 15);
        CompiledOperatorTable grouped_operators;
        grouped_operators.bind_weight_count(grouped_weights.size());
        ActivationBuffer batch(65, width);
        for (size_t i = 0; i < batch.values().size(); ++i)
            batch.row(0)[i] = static_cast<float>(static_cast<int>((i * 13) % 37) - 18) / 64.0f;

        const auto check_cache = [&](const LayerCache& actual, const LayerCache& expected) {
            check(actual.gated_delta_token_count == expected.gated_delta_token_count);
            check(actual.gated_delta_convolution.size() == expected.gated_delta_convolution.size());
            check(actual.gated_delta_recurrent.size() == expected.gated_delta_recurrent.size());
            for (size_t i = 0; i < actual.gated_delta_convolution.size(); ++i)
                check_near(actual.gated_delta_convolution[i], expected.gated_delta_convolution[i], 1e-5f);
            for (size_t i = 0; i < actual.gated_delta_recurrent.size(); ++i)
                check_near(actual.gated_delta_recurrent[i], expected.gated_delta_recurrent[i], 1e-5f);
        };
        for (uint32_t flags : {uint32_t{0}, uint32_t{AttentionBlockSigmoidGate | AttentionBlockExternalResidual}})
        {
            grouped.flags = flags;
            thread_limit.set(1);
            LayerCache reference;
            LayerCache prefix;
            GatedDeltaScratch row_scratch;
            ActivationBuffer expected(batch.rows(), width);
            ActivationBuffer token(1, width);
            ActivationBuffer row_output;
            for (size_t row = 0; row < batch.rows(); ++row)
            {
                std::copy_n(batch.row(row), width, token.row(0));
                check(static_cast<bool>(forward_gated_delta(grouped_weights, grouped_operators, grouped, ExecutionBackend::Cpu,
                                                            reference, row_scratch, token, row_output, g_test_optimization_flags)));
                std::copy_n(row_output.row(0), width, expected.row(row));
                if (row == 32)
                    prefix = reference;
            }
            GatedDeltaScratch batch_scratch;
            for (uint32_t threads : {1u, 3u, 8u})
            {
                thread_limit.set(threads);
                LayerCache batch_cache;
                ActivationBuffer actual = batch;
                check(static_cast<bool>(forward_gated_delta(grouped_weights, grouped_operators, grouped, ExecutionBackend::Cpu,
                                                            batch_cache, batch_scratch, actual, actual, g_test_optimization_flags)));
                for (size_t i = 0; i < actual.values().size(); ++i)
                    check_near(actual.values()[i], expected.values()[i], 1e-5f);
                check_cache(batch_cache, reference);

                std::array<LayerCache, 1> transaction;
                GatedDeltaScratch transaction_scratch;
                check(static_cast<bool>(begin_state_cache_transaction(transaction, batch.rows())));
                check(static_cast<bool>(forward_gated_delta(grouped_weights, grouped_operators, grouped, ExecutionBackend::Cpu,
                                                            transaction.front(), transaction_scratch, batch, actual, g_test_optimization_flags)));
                check(static_cast<bool>(finish_state_cache_transaction(transaction, 33)));
                check_cache(transaction.front(), prefix);

                // Continue after rollback, mixing a batch and a final decode.
                ActivationBuffer tail(31, width);
                std::copy_n(batch.row(33), tail.values().size(), tail.row(0));
                check(static_cast<bool>(forward_gated_delta(grouped_weights, grouped_operators, grouped, ExecutionBackend::Cpu,
                                                            transaction.front(), transaction_scratch, tail, actual, g_test_optimization_flags)));
                std::copy_n(batch.row(64), width, token.row(0));
                check(static_cast<bool>(forward_gated_delta(grouped_weights, grouped_operators, grouped, ExecutionBackend::Cpu,
                                                            transaction.front(), transaction_scratch, token, actual, g_test_optimization_flags)));
                for (size_t i = 0; i < width; ++i)
                    check_near(actual.row(0)[i], expected.row(64)[i], 1e-5f);
                check_cache(transaction.front(), reference);
            }
        }
    }
}

} // namespace moe
} // namespace ncnn
