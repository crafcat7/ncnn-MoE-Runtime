#include "test_runtime_decls.h"
#include "test_runtime_support.h"

namespace ncnn {
namespace moe {

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
        layer.moe.expert_count = expert_count;
        layer.moe.top_k = 4;
        layer.moe.intermediate_size = 2880;
        layer.moe.expert_weight_dtype = DType::MxFp4;
        layer.moe.flags |= MoeDescriptorRouterBias | MoeDescriptorProjectionBias;
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

    auto available_limited_plan = plan_model_memory(large,
                                                    options,
                                                    physical_memory,
                                                    false,
                                                    12 * gibibyte);
    check(static_cast<bool>(available_limited_plan));
    check(static_cast<bool>(available_limited_plan.value().host_memory_budget == 10 * gibibyte));
    check(static_cast<bool>(available_limited_plan.value().expert_cache_size
                            == 10 * gibibyte - available_limited_plan.value().estimated_dense_size));

    constexpr uint64_t staging_per_device = 64 * 1024 * 1024;
    Option hybrid_options = options;
    hybrid_options.hybrid_mode = HybridMode::HybridExperts;
    auto hybrid_limited_plan = plan_model_memory(large,
                                                 hybrid_options,
                                                 physical_memory,
                                                 false,
                                                 12 * gibibyte);
    check(static_cast<bool>(hybrid_limited_plan));
    const uint64_t limited_dense_size = hybrid_limited_plan.value().estimated_dense_size;
    check(static_cast<bool>(hybrid_limited_plan.value().expert_cache_size
                            == 10 * gibibyte - limited_dense_size - staging_per_device));
    ModelMemoryPlan finalized_hybrid_plan = hybrid_limited_plan.value();
    check(static_cast<bool>(finalize_model_memory_plan(finalized_hybrid_plan, hybrid_options, 0)));
    const uint64_t finalized_cache_size = finalized_hybrid_plan.expert_cache_size;
    check(static_cast<bool>(finalize_model_memory_plan(finalized_hybrid_plan, hybrid_options, 0)));
    check(static_cast<bool>(finalized_hybrid_plan.expert_cache_size == finalized_cache_size));

    Option small_gpu_cache = hybrid_options;
    small_gpu_cache.expert_gpu_cache_size = 32 * 1024 * 1024;
    auto small_gpu_cache_plan = plan_model_memory(large,
                                                  small_gpu_cache,
                                                  physical_memory,
                                                  false,
                                                  12 * gibibyte);
    check(static_cast<bool>(small_gpu_cache_plan));
    check(static_cast<bool>(small_gpu_cache_plan.value().expert_cache_size
                            == 10 * gibibyte - limited_dense_size - small_gpu_cache.expert_gpu_cache_size));

    Option capped_gpu_cache = hybrid_options;
    capped_gpu_cache.expert_gpu_cache_size = 128 * 1024 * 1024;
    auto capped_gpu_cache_plan = plan_model_memory(large,
                                                   capped_gpu_cache,
                                                   physical_memory,
                                                   false,
                                                   12 * gibibyte);
    check(static_cast<bool>(capped_gpu_cache_plan));
    check(static_cast<bool>(capped_gpu_cache_plan.value().expert_cache_size
                            == 10 * gibibyte - limited_dense_size - staging_per_device));
    capped_gpu_cache.vulkan_device_indices = {0, 1};
    capped_gpu_cache.expert_gpu_cache_size = 256 * 1024 * 1024;
    auto multi_gpu_cache_plan = plan_model_memory(large,
                                                  capped_gpu_cache,
                                                  physical_memory,
                                                  false,
                                                  12 * gibibyte);
    check(static_cast<bool>(multi_gpu_cache_plan));
    check(static_cast<bool>(multi_gpu_cache_plan.value().expert_cache_size
                            == 10 * gibibyte - limited_dense_size - 2 * staging_per_device));

    Option cpu_only_cache = hybrid_options;
    cpu_only_cache.hybrid_mode = HybridMode::CpuOnly;
    auto cpu_only_plan = plan_model_memory(large,
                                           cpu_only_cache,
                                           physical_memory,
                                           false,
                                           12 * gibibyte);
    check(static_cast<bool>(cpu_only_plan));
    check(static_cast<bool>(cpu_only_plan.value().expert_cache_size
                            == 10 * gibibyte - limited_dense_size));
    Option disabled_gpu_cache = hybrid_options;
    disabled_gpu_cache.flags |= OptionDisableGpuExpertExecution;
    auto disabled_gpu_plan = plan_model_memory(large,
                                               disabled_gpu_cache,
                                               physical_memory,
                                               false,
                                               12 * gibibyte);
    check(static_cast<bool>(disabled_gpu_plan));
    check(static_cast<bool>(disabled_gpu_plan.value().expert_cache_size
                            == 10 * gibibyte - limited_dense_size));

    const uint64_t active_expert_size = hybrid_limited_plan.value().minimum_active_expert_size;
    Option cpu_tight_cache = cpu_only_cache;
    cpu_tight_cache.expert_memory_mode = ExpertMemoryMode::OnDemand;
    cpu_tight_cache.expert_cache_size = active_expert_size;
    cpu_tight_cache.host_memory_budget = limited_dense_size + 4 * gibibyte + active_expert_size - 1;
    auto cpu_tight_plan = plan_model_memory(large, cpu_tight_cache, physical_memory);
    check(static_cast<bool>(!cpu_tight_plan));
    check(static_cast<bool>(cpu_tight_plan.error().code == ErrorCode::InvalidArgument));

    Option explicit_cpu_cache = cpu_tight_cache;
    explicit_cpu_cache.expert_cache_size = 4 * gibibyte + active_expert_size;
    explicit_cpu_cache.host_memory_budget = limited_dense_size + explicit_cpu_cache.expert_cache_size;
    auto explicit_cpu_plan = plan_model_memory(large, explicit_cpu_cache, physical_memory);
    check(static_cast<bool>(explicit_cpu_plan));
    check(static_cast<bool>(explicit_cpu_plan.value().expert_cache_size == explicit_cpu_cache.expert_cache_size));
    ModelMemoryPlan finalized_explicit_cpu_plan = explicit_cpu_plan.value();
    check(static_cast<bool>(finalize_model_memory_plan(finalized_explicit_cpu_plan, explicit_cpu_cache, 0)));
    check(static_cast<bool>(finalized_explicit_cpu_plan.expert_cache_size == explicit_cpu_cache.expert_cache_size));

    Option exact_staging_budget = hybrid_options;
    exact_staging_budget.expert_memory_mode = ExpertMemoryMode::OnDemand;
    exact_staging_budget.expert_cache_size = active_expert_size;
    const uint64_t exact_host_budget = limited_dense_size + staging_per_device + active_expert_size;
    uint64_t exact_available_memory = 2 * gibibyte + exact_host_budget;
    auto exact_staging_plan = plan_model_memory(large,
                                                exact_staging_budget,
                                                physical_memory,
                                                false,
                                                exact_available_memory);
    check(static_cast<bool>(exact_staging_plan));
    check(static_cast<bool>(exact_staging_plan.value().host_memory_budget == exact_host_budget));
    check(static_cast<bool>(exact_staging_plan.value().expert_cache_size == active_expert_size));
    ModelMemoryPlan finalized_exact_plan = exact_staging_plan.value();
    check(static_cast<bool>(finalize_model_memory_plan(finalized_exact_plan, exact_staging_budget, 0)));
    check(static_cast<bool>(finalized_exact_plan.expert_cache_size == active_expert_size));
    --exact_available_memory;
    auto undersized_staging_plan = plan_model_memory(large,
                                                     exact_staging_budget,
                                                     physical_memory,
                                                     false,
                                                     exact_available_memory);
    check(static_cast<bool>(!undersized_staging_plan));
    check(static_cast<bool>(undersized_staging_plan.error().code == ErrorCode::InvalidArgument));

    MoeModelDescriptor qnk = gpt_oss_memory_descriptor(1, 24);
    qnk.vocabulary_size = 128;
    qnk.hidden_size = 4096;
    qnk.intermediate_size = 4096;
    qnk.layers.front().moe.intermediate_size = 4096;
    qnk.layers.front().moe.expert_weight_dtype = DType::Q4K;
    Option qnk_options;
    qnk_options.host_memory_budget = 3 * gibibyte;
    auto qnk_raw_plan = plan_model_memory(qnk, qnk_options, 4 * gibibyte);
    check(static_cast<bool>(qnk_raw_plan));
    check(static_cast<bool>(qnk_raw_plan.value().estimated_cpu_packed_expert_size == 0));
    auto qnk_packed_plan = plan_model_memory(qnk,
                                             qnk_options,
                                             4 * gibibyte,
                                             false,
                                             0,
                                             true);
    check(static_cast<bool>(!qnk_packed_plan));
    check(static_cast<bool>(qnk_packed_plan.error().code == ErrorCode::InvalidArgument));

    auto available_plan = plan_model_memory(large,
                                            options,
                                            physical_memory,
                                            false,
                                            16 * gibibyte);
    check(static_cast<bool>(available_plan));
    check(static_cast<bool>(available_plan.value().available_memory_size == 16 * gibibyte));
    check(static_cast<bool>(available_plan.value().host_memory_budget == 14 * gibibyte));

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

    MoeModelDescriptor native = gpt_oss_memory_descriptor(1, 2);
    native.hidden_size = 256;
    native.intermediate_size = 256;
    native.experts_per_token = 1;
    native.layers.front().moe.intermediate_size = 256;
    native.layers.front().moe.top_k = 1;
    native.layers.front().moe.layout = ExpertLayout::GateUpDown;
    native.layers.front().moe.expert_weight_dtype = DType::Q4K;
    const uint64_t gate_size = qnk_storage_bytes(DType::Q4K, 256, 256);
    const uint64_t down_size = qnk_storage_bytes(DType::Q6K, 256, 256);
    const uint64_t raw_pair_size = gate_size * 2 + down_size;
    const uint64_t map_pair_size = uint64_t{3} * 256 * sizeof(uint32_t);
    const uint64_t bias_pair_size = uint64_t{3} * 256 * sizeof(uint16_t);
    const uint64_t resident_pair_size = raw_pair_size + map_pair_size * 2;
    const uint64_t fixed_size = 8 * sizeof(float) + 2 * (map_pair_size + bias_pair_size);
    WeightMapping native_weights;
    TensorData dense;
    dense.dtype = DType::Float32;
    dense.shape = {8};
    dense.float32_data.assign(8, 1.0f);
    native_weights.emplace("embedding.weight", std::move(dense));
    uint64_t offset = 0;
    for (uint32_t expert_id = 0; expert_id < 2; ++expert_id)
    {
        const std::string prefix = "layers.0.experts." + std::to_string(expert_id) + ".";
        for (const char* projection : {"gate", "up", "down"})
        {
            TensorData weight;
            weight.dtype = std::string_view(projection) == "down" ? DType::Q6K : DType::Q4K;
            weight.shape = {256, 256};
            weight.input_column_indices.resize(256);
            for (uint32_t column = 0; column < 256; ++column)
                weight.input_column_indices[column] = 255 - column;
            auto storage = std::make_shared<MxFp4FileStorage>();
            storage->blocks_path = "memory-plan-qnk.bin";
            storage->blocks_offset = offset;
            storage->blocks_size = qnk_storage_bytes(weight.dtype, 256, 256);
            offset += storage->blocks_size;
            weight.mxfp4_file_storage = std::move(storage);
            native_weights.emplace(prefix + projection + ".weight", std::move(weight));
            TensorData bias;
            bias.dtype = DType::BFloat16;
            bias.shape = {256};
            bias.bfloat16_data.assign(256, 0);
            native_weights.emplace(prefix + projection + ".bias", std::move(bias));
        }
    }

    Option native_options;
    native_options.expert_memory_mode = ExpertMemoryMode::OnDemand;
    native_options.expert_cache_size = resident_pair_size;
    native_options.host_memory_budget = 2 * gibibyte + fixed_size + resident_pair_size;
    auto native_plan = plan_model_memory(native, native_options, 4 * gibibyte, false, 0, false, &native_weights);
    check(static_cast<bool>(native_plan));
    check(native_plan.value().selected_mode == ExpertMemoryMode::OnDemand);
    check(native_plan.value().estimated_dense_size == fixed_size);
    check(native_plan.value().expert_pair_resident_size == resident_pair_size);
    check(native_plan.value().minimum_active_expert_size == resident_pair_size);
    check(native_plan.value().expert_cache_size == resident_pair_size);

    {
        MoeModelDescriptor qwen = native;
        qwen.hyper_connection_kind = HyperConnectionKind::GatedResidual;
        qwen.layers.front().attention.kind = AttentionKind::GatedDeltaNet;
        qwen.layers.push_back(qwen.layers.front());
        qwen.layers.back().attention.kind = AttentionKind::Standard;
        qwen.layers.back().attention.flags = AttentionDescriptorQsa | AttentionDescriptorQueryKeyNorm;
        qwen.layers.front().moe.shared_expert_count = 1;
        WeightMapping qwen_weights = native_weights;
        for (const auto& entry : native_weights)
        {
            if (entry.first.find("layers.0.experts.") == 0)
                qwen_weights.emplace("layers.1." + entry.first.substr(std::string("layers.0.").size()), entry.second);
        }
        WeightMapping weights = qwen_weights;
        uint64_t expected_release = 0;
        const auto add_owned = [&](const std::string& name, DType dtype, uint32_t rows, uint32_t columns, bool released) {
            TensorData tensor;
            tensor.dtype = dtype;
            tensor.shape = {rows, columns};
            uint64_t payload_size = 0;
            if (dtype == DType::Float32)
            {
                tensor.float32_data.assign(rows * columns, 1.0f);
                payload_size = tensor.float32_data.size() * sizeof(float);
            }
            else if (dtype == DType::BFloat16)
            {
                tensor.bfloat16_data.assign(rows * columns, float_to_bfloat16(1.0f));
                payload_size = tensor.bfloat16_data.size() * sizeof(uint16_t);
            }
            else
            {
                tensor.quantized_data.assign(qnk_storage_bytes(dtype, rows, columns), 0);
                payload_size = tensor.quantized_data.size();
            }
            // These allocations survive even if the primary payload is freed.
            tensor.input_column_indices.assign(3, 0);
            tensor.quantization_scales.assign(2, 1.0f);
            weights.insert_or_assign(name, std::move(tensor));
            if (released)
                expected_release += payload_size;
        };
        add_owned("lm_head.weight", DType::BFloat16, 5, 32, true);
        add_owned("layers.0.attention.delta.qkv.weight", DType::Float32, 7, 32, true);
        add_owned("layers.0.attention.delta.z.weight", DType::Q8_0, 3, 32, true);
        add_owned("layers.0.attention.delta.beta.weight", DType::Q4K, 2, 256, false);
        add_owned("layers.0.attention.delta.alpha.weight", DType::Q6K, 2, 256, false);
        add_owned("layers.0.attention.output.weight", DType::Q5_1, 3, 32, false);
        add_owned("layers.1.attention.query_norm.weight", DType::BFloat16, 1, 32, true);
        add_owned("layers.1.attention.key_norm.weight", DType::Float32, 1, 32, true);
        for (const char* projection : {"gate", "up", "down"})
            add_owned("layers.0.shared_expert." + std::string(projection) + ".weight", DType::Q8_0, 2, 32, true);
        for (const char* name : {"layers.0.attention.delta.norm.weight", "layers.0.hc.attention.base",
                                 "layers.0.ple.input_projection.weight", "layers.0.shared_expert.router_gate.weight",
                                 "layers.1.attention.query.weight", "layers.1.attention.output.weight",
                                 "layers.1.attention.qsa.query_norm.weight", "layers.1.attention.qsa.key_norm.weight"})
            add_owned(name, DType::Float32, 2, 32, false);
        add_owned("prefix.lm_head.weight", DType::BFloat16, 1, 32, false);

        Option hybrid = native_options;
        hybrid.hybrid_mode = HybridMode::HybridExperts;
        hybrid.host_memory_budget = 3 * gibibyte;
        auto retained = plan_model_memory(qwen, hybrid, 4 * gibibyte, false, 0, false, &weights);
        auto predicted = plan_model_memory(qwen, hybrid, 4 * gibibyte, true, 0, false, &weights, true);
        check(retained && predicted);
        check(retained.value().estimated_vulkan_dense_host_release_size == 0);
        check(predicted.value().estimated_vulkan_dense_host_release_size == expected_release);
        check(predicted.value().estimated_dense_size + expected_release == retained.value().estimated_dense_size);
        const uint64_t full_dense_size = retained.value().estimated_dense_size;

        // Available-memory admission reserves idle staging before compilation,
        // then settles dense storage without changing that budget.
        Option tight = hybrid;
        tight.host_memory_budget = 0;
        const uint64_t budget_without_staging = predicted.value().estimated_dense_size + tight.expert_cache_size;
        const uint64_t budget = budget_without_staging + staging_per_device;
        const uint64_t available = 2 * gibibyte + budget;
        const uint64_t available_without_staging = 2 * gibibyte + budget_without_staging;
        auto admitted = plan_model_memory(qwen, tight, 4 * gibibyte, true, available, false, &weights, true);
        check(admitted && admitted.value().host_memory_budget == budget);
        check(admitted.value().estimated_dense_size == predicted.value().estimated_dense_size);
        check(static_cast<bool>(finalize_model_memory_plan(admitted.value(), tight, expected_release)));
        check(admitted.value().estimated_dense_size == full_dense_size - expected_release);
        check(admitted.value().expert_cache_size == tight.expert_cache_size);
        auto missing_dense = plan_model_memory(qwen, tight, 4 * gibibyte, false, available, false, &weights, true);
        auto missing_attention = plan_model_memory(qwen, tight, 4 * gibibyte, true, available, false, &weights);
        check(!missing_dense && !missing_attention);
        Option cpu = tight;
        cpu.hybrid_mode = HybridMode::CpuOnly;
        check(!plan_model_memory(qwen, cpu, 4 * gibibyte, true, available_without_staging, false, &weights, true));
        cpu = tight;
        cpu.flags |= OptionDisableGpuExpertExecution;
        check(!plan_model_memory(qwen, cpu, 4 * gibibyte, true, available_without_staging, false, &weights, true));
        for (uint64_t disabled : {static_cast<uint64_t>(OptimizationVulkanAttention), static_cast<uint64_t>(OptimizationVulkanQnK),
                                  static_cast<uint64_t>(OptimizationVulkanExpertGpuPriority), static_cast<uint64_t>(OptimizationVulkanCommandGraph)})
        {
            Option legacy = tight;
            legacy.optimization_flags &= ~disabled;
            check(!plan_model_memory(qwen, legacy, 4 * gibibyte, true, available, false, &weights, true));
        }
        ModelMemoryPlan partial = predicted.value();
        check(static_cast<bool>(finalize_model_memory_plan(partial, hybrid, expected_release / 2)));
        check(partial.estimated_dense_size == full_dense_size - expected_release / 2);
        check(partial.estimated_vulkan_dense_host_release_size == expected_release / 2);
        ModelMemoryPlan failed_release = admitted.value();
        check(!finalize_model_memory_plan(failed_release, tight, 0));

        // Native blocks other than Q8_0 keep their original payload, including
        // when they occupy a handle which otherwise has a resident operator.
        for (DType dtype : {DType::Q4K, DType::Q5K, DType::Q6K, DType::IQ4NL, DType::Q5_1})
        {
            WeightMapping native_head = qwen_weights;
            native_head.emplace("lm_head.weight", weights.at("layers.0.attention.delta.beta.weight"));
            TensorData& head = native_head.at("lm_head.weight");
            head.dtype = dtype;
            head.quantized_data.assign(qnk_storage_bytes(dtype, 2, 256), 0);
            auto unchanged = plan_model_memory(qwen, hybrid, 4 * gibibyte, true, 0, false, &native_head, true);
            check(unchanged && unchanged.value().estimated_vulkan_dense_host_release_size == 0);
        }

        ScopedTestDirectory directory("ncnn_moe_qwen_release_plan_");
        const auto path = directory.path() / "norms.bin";
        const std::vector<float> payload(32, 1.0f);
        {
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size() * sizeof(float)));
            check(static_cast<bool>(file));
        }
        const uint64_t mapped_size = payload.size() * sizeof(float);
        auto mapping = MappedFileRange::open(path, 0, mapped_size);
        check(static_cast<bool>(mapping));
        TensorData mapped;
        mapped.dtype = DType::Float32;
        mapped.shape = {32};
        mapped.mapped_data = mapping.value()->share_data();
        mapped.mapped_size = mapped_size;
        WeightMapping mapped_weights = qwen_weights;
        mapped_weights.emplace("layers.1.attention.query_norm.weight", mapped);
        mapped_weights.emplace("layers.1.attention.key_norm.weight", mapped);
        auto full_owner = plan_model_memory(qwen, hybrid, 4 * gibibyte, true, 0, false, &mapped_weights, true);
        check(full_owner && full_owner.value().estimated_vulkan_dense_host_release_size == mapped_size);
        mapped_weights.emplace("retained.alias.weight", mapped);
        auto tied_owner = plan_model_memory(qwen, hybrid, 4 * gibibyte, true, 0, false, &mapped_weights, true);
        check(tied_owner && tied_owner.value().estimated_vulkan_dense_host_release_size == 0);
    }

    {
        ScopedTestDirectory directory("ncnn_moe_ple_memory_plan_");
        const auto path = directory.path() / "ple.bin";
        const std::vector<uint8_t> payload(16 * 18, 0x53);
        {
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
            check(static_cast<bool>(file));
        }
        auto mapping = MappedFileRange::open(path, 0, payload.size());
        check(static_cast<bool>(mapping));
        TensorData table;
        table.dtype = DType::IQ4NL;
        table.shape = {16, 32};
        table.mapped_data = mapping.value()->share_data();
        table.mapped_size = payload.size();
        MoeModelDescriptor ple_native = native;
        PleDescriptor& ple = ple_native.layers.front().ple;
        ple.embedding_dimension = 256;
        ple.ngram_size = 3;
        ple.heads_per_ngram = 4;
        ple.convolution_kernel_size = 1;
        ple.embedding_shard_count = 2;
        ple.embedding_row_count = 32;
        const std::string shard_name = "layers.0.ple.embedding_shard.0";
        const auto plan_with = [&](const std::string& name, const TensorData& tensor, const MoeModelDescriptor& model, const Option& opt) {
            WeightMapping weights = native_weights;
            weights.emplace(name, tensor);
            return plan_model_memory(model, opt, 4 * gibibyte, false, 0, false, &weights);
        };
        auto paged = plan_with(shard_name, table, ple_native, native_options);
        check(static_cast<bool>(paged));
        check(paged.value().estimated_dense_size == fixed_size);
        check(paged.value().expert_cache_size == resident_pair_size);

        TensorData owned = table;
        owned.mapped_data.reset();
        owned.mapped_size = 0;
        owned.quantized_data = payload;
        auto rejected_owned = plan_with(shard_name, owned, ple_native, native_options);
        check(!rejected_owned && rejected_owned.error().code == ErrorCode::InvalidArgument);
        Option extra_budget = native_options;
        extra_budget.host_memory_budget += payload.size();
        auto resident = plan_with(shard_name, owned, ple_native, extra_budget);
        check(resident && resident.value().estimated_dense_size == fixed_size + payload.size());
        for (const std::string& name : {std::string("ordinary.embedding.weight"),
                                        std::string("prefix.layers.0.ple.embedding_shard.0"),
                                        std::string("layers.0.ple.embedding_shard.0.extra"),
                                        std::string("layers.00.ple.embedding_shard.0"),
                                        std::string("layers.0.ple.embedding_shard.00"),
                                        std::string("layers.0.ple.embedding_shard.2"),
                                        std::string("layers.1.ple.embedding_shard.0")})
        {
            auto ordinary = plan_with(name, table, ple_native, extra_budget);
            check(ordinary && ordinary.value().estimated_dense_size == fixed_size + payload.size());
        }
        MoeModelDescriptor undeclared = ple_native;
        undeclared.layers.front().ple = {};
        auto not_ple = plan_with(shard_name, table, undeclared, extra_budget);
        check(not_ple && not_ple.value().estimated_dense_size == fixed_size + payload.size());

        TensorData mixed = table;
        mixed.quantized_data = payload;
        mixed.quantization_scales.assign(3, 1.0f);
        mixed.input_column_indices.resize(32);
        for (uint32_t column = 0; column < 32; ++column)
            mixed.input_column_indices[column] = column;
        const uint64_t owned_size = payload.size() + 3 * sizeof(float) + 32 * sizeof(uint32_t);
        extra_budget.host_memory_budget = native_options.host_memory_budget + owned_size;
        auto mixed_plan = plan_with(shard_name, mixed, ple_native, extra_budget);
        check(mixed_plan && mixed_plan.value().estimated_dense_size == fixed_size + owned_size);
    }

    Option short_cache = native_options;
    --short_cache.expert_cache_size;
    auto short_cache_plan = plan_model_memory(native, short_cache, 4 * gibibyte, false, 0, false, &native_weights);
    check(!short_cache_plan && short_cache_plan.error().code == ErrorCode::InvalidArgument);
    Option short_budget = native_options;
    --short_budget.host_memory_budget;
    auto short_budget_plan = plan_model_memory(native, short_budget, 4 * gibibyte, false, 0, false, &native_weights);
    check(!short_budget_plan && short_budget_plan.error().code == ErrorCode::InvalidArgument);

    native.layers.front().moe.expert_weight_dtype = DType::BFloat16;
    auto representative_plan = plan_model_memory(native, native_options, 4 * gibibyte, false, 0, false, &native_weights);
    check(static_cast<bool>(representative_plan));
    check(representative_plan.value().selected_mode == ExpertMemoryMode::OnDemand);
    check(representative_plan.value().estimated_dense_size == fixed_size);
    check(representative_plan.value().expert_pair_resident_size == resident_pair_size);

    TensorData& resident_gate = native_weights.at("layers.0.experts.0.gate.weight");
    resident_gate.mxfp4_file_storage.reset();
    resident_gate.quantized_data.resize(static_cast<size_t>(gate_size));
    auto incomplete_plan = plan_model_memory(native, native_options, 4 * gibibyte, false, 0, false, &native_weights);
    check(!incomplete_plan && incomplete_plan.error().code == ErrorCode::UnsupportedModel);
    native_options.expert_memory_mode = ExpertMemoryMode::Auto;
    native_options.expert_cache_size = 0;
    native_options.host_memory_budget = 3 * gibibyte;
    auto native_eager_plan = plan_model_memory(native, native_options, 4 * gibibyte, false, 0, false, &native_weights);
    check(static_cast<bool>(native_eager_plan));
    check(native_eager_plan.value().selected_mode == ExpertMemoryMode::Eager);
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

    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> tie_across_8(18, -2.0f);
    tie_across_8[7] = 9.0f;
    tie_across_8[8] = 9.0f;
    tie_across_8[17] = 9.0f;
    std::vector<float> tie_across_16(33, -2.0f);
    tie_across_16[15] = 11.0f;
    tie_across_16[16] = 11.0f;
    tie_across_16[32] = 11.0f;
    std::vector<float> nonfinite_and_tail_tie(19, -infinity);
    nonfinite_and_tail_tie[0] = nan;
    nonfinite_and_tail_tie[7] = infinity;
    nonfinite_and_tail_tie[8] = 5.0f;
    nonfinite_and_tail_tie[17] = 5.0f;
    std::vector<float> signed_zero_tie(17, 0.0f);
    signed_zero_tie[0] = -0.0f;
    signed_zero_tie[8] = -0.0f;
    signed_zero_tie[16] = -0.0f;
    const float denormal = std::numeric_limits<float>::denorm_min();
    std::vector<float> subnormal_tie(17, -1.0f);
    subnormal_tie[0] = -denormal;
    subnormal_tie[7] = denormal;
    subnormal_tie[8] = denormal;
    subnormal_tie[16] = denormal;
    struct GreedyCase
    {
        const std::vector<float>& logits;
        int32_t token_id;
    };
    const std::array<GreedyCase, 5> greedy_cases = {{
        {tie_across_8, 7},
        {tie_across_16, 15},
        {nonfinite_and_tail_tie, 8},
        {signed_zero_tie, 0},
        {subnormal_tie, 7},
    }};
    for (const GreedyCase& test_case : greedy_cases)
    {
        const auto sampled = session.value()->sample(test_case.logits, greedy_options);
        check(static_cast<bool>(sampled));
        check(static_cast<bool>(sampled.value().token_id == test_case.token_id));
        check_near(sampled.value().probability, 1.0f, 1e-6f);
    }

    logits = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    greedy = session.value()->sample(logits, greedy_options);
    check(static_cast<bool>(!greedy));
    check(static_cast<bool>(greedy.error().code == ErrorCode::InvalidArgument));
    std::vector<float> all_nonfinite(33, -infinity);
    for (size_t index = 0; index < all_nonfinite.size(); ++index)
    {
        if (index % 3 == 0)
            all_nonfinite[index] = nan;
        else if (index % 3 == 1)
            all_nonfinite[index] = infinity;
        else
            all_nonfinite[index] = -infinity;
    }
    const auto vector_invalid = session.value()->sample(all_nonfinite, greedy_options);
    check(static_cast<bool>(!vector_invalid));
    check(static_cast<bool>(vector_invalid.error().code == ErrorCode::InvalidArgument));

    logits = {1.0f, 2.0f, 3.0f};
    SamplingOptions top_k_options;
    top_k_options.top_k = 1;
    auto top_k = session.value()->sample(logits, top_k_options);
    check(static_cast<bool>(top_k));
    check(static_cast<bool>(top_k.value().token_id == 2));

    // Leave filtered token ids in the Session scratch, then ensure default
    // sampling still uses the complete contiguous vocabulary.
    logits = {5.0f, 5.0f, 5.0f};
    SamplingOptions default_options;
    auto default_after_filter = session.value()->sample(logits, default_options);
    check(static_cast<bool>(default_after_filter));
    check_near(default_after_filter.value().probability, 1.0f / 3.0f, 1e-6f);

    std::vector<float> wide_logits(33, -infinity);
    wide_logits[0] = 1.0f;
    wide_logits[1] = nan;
    wide_logits[16] = 2.0f;
    wide_logits[17] = infinity;
    wide_logits[32] = 3.0f;
    const float wide_normalizer = std::exp(-2.0f) + std::exp(-1.0f) + 1.0f;
    const auto check_wide_sample = [&wide_logits, &wide_normalizer](const Result<SampledToken>& sampled) {
        check(static_cast<bool>(sampled));
        if (!sampled)
            return;
        const int32_t token_id = sampled.value().token_id;
        check(token_id == 0 || token_id == 16 || token_id == 32);
        const float weight = std::exp(wide_logits[token_id] - 3.0f);
        check_near(sampled.value().probability, weight / wide_normalizer, weight / wide_normalizer * 1e-3f + 1e-6f);
    };
    check_wide_sample(session.value()->sample(wide_logits));
    SamplingOptions wide_top_k_options;
    wide_top_k_options.top_k = 2;
    const auto wide_filtered = session.value()->sample(wide_logits, wide_top_k_options);
    check(static_cast<bool>(wide_filtered));
    check(wide_filtered.value().token_id == 16 || wide_filtered.value().token_id == 32);
    check_wide_sample(session.value()->sample(wide_logits));

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

    // Top-p without top-k grows a bounded candidate set until the retained
    // mass reaches top_p, then truncates that prefix.
    std::vector<float> flat_logits(200, 0.0f);
    SamplingOptions flat_options;
    for (const auto& item : std::vector<std::pair<float, float>>{{0.5f, 0.01f}, {0.999f, 0.005f}})
    {
        flat_options.top_p = item.first;
        const auto flat = session.value()->sample(flat_logits, flat_options);
        check(static_cast<bool>(flat));
        check_near(flat.value().probability, item.second, 1e-3f);
    }

    SamplingOptions invalid_options;
    invalid_options.top_p = 0.0f;
    auto invalid = session.value()->sample(logits, invalid_options);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));
    const std::span<const float> empty_logits;
    auto empty_invalid = session.value()->sample(empty_logits, invalid_options);
    check(static_cast<bool>(!empty_invalid));
    check(static_cast<bool>(empty_invalid.error().message == "cannot sample empty logits"));

    GenerationOptions generation_options;
    generation_options.max_new_tokens = 3;
    generation_options.sampling.temperature = 0.0f;
    std::vector<StreamToken> streamed;
    auto generated = session.value()->generate(std::vector<int32_t>{0}, generation_options, [&streamed, &session](const StreamToken& token) {
            check(session.value()->statistics().prefill_tokens == 1);
            streamed.push_back(token);
            return true; }, [](int32_t token_id) { return "<" + std::to_string(token_id) + ">"; });
    check(static_cast<bool>(generated));
    check(static_cast<bool>(generated.value().tokens.size() == 3));
    check(static_cast<bool>(streamed.size() == 3));
    check(static_cast<bool>(generated.value().tokens[0].text == "<" + std::to_string(generated.value().tokens[0].token_id) + ">"));
    check(static_cast<bool>(session.value()->sequence_length() == 3));

    auto mutable_options_session = runtime.create_session(model.value(), session_options);
    check(static_cast<bool>(mutable_options_session));
    GenerationOptions mutable_options;
    mutable_options.max_new_tokens = 3;
    mutable_options.sampling.temperature = 0.0f;
    auto invalidated_options = mutable_options_session.value()->generate(std::vector<int32_t>{0}, mutable_options,
                                                                         [&mutable_options](const StreamToken& token) {
                                                                             if (token.index == 0)
                                                                                 mutable_options.sampling.top_p = 0.0f;
                                                                             return true;
                                                                         });
    check(static_cast<bool>(!invalidated_options));
    check(static_cast<bool>(invalidated_options.error().code == ErrorCode::InvalidArgument));
    check(static_cast<bool>(invalidated_options.error().message == "top_p must be in the range (0, 1]"));

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

void test_batch_workspace_reuse()
{
    TemporaryModelPackage moe_package;
    AttentionPackage attention_package(false, 0, false, false);
    Bfloat16StagedBatchPackage large_package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    const auto moe_model = runtime.load_model(moe_package.path(), opt).value();
    const auto attention_model = runtime.load_model(attention_package.path(), opt).value();
    const auto large_model = runtime.load_model(large_package.path(), opt).value();
    BatchWorkspace workspace;
    const auto check_released = [&] {
        check(workspace.expert.cache_leases.empty());
        check(workspace.expert.cache_requests.empty());
        check(workspace.expert.backend_requests.empty());
        check(workspace.expert.decode_tasks.empty());
        check(workspace.expert.kernels.effective_tasks.empty());
        check(workspace.attention_batch_entries.empty());
        check(workspace.gated_delta_entries.empty());
        check(workspace.gated_delta_device_entries.empty());
        check(workspace.staged_attention_caches.empty());
        check(workspace.attention.latent_row_contexts.empty());
        check(workspace.attention.latent_caches.empty());
        check(workspace.attention.latent_projected_compressed_counts.empty());
        for (const auto& active : workspace.staged_state.expert_slots)
        {
            check(!active.lease.gate_up && !active.lease.down);
            check(!active.lease.gate_up_operator && !active.lease.down_operator);
        }
    };
    const auto run_model = [&](const ModelPtr& model) {
        const CompiledModel& compiled = model_compiled(*model);
        CompiledModel execution_model = compiled;
        const bool force_cpu_prefetch = model.get() == moe_model.get();
        if (force_cpu_prefetch)
        {
            bool found_expert_group = false;
            for (Layer& node : execution_model.net.layers)
            {
                if (node.kind != LayerKind::ExpertGroup)
                    continue;
                check(node.backend == ExecutionBackend::Cpu);
                node.flags |= LayerCpuPrefetch;
                found_expert_group = true;
            }
            check(found_expert_group);
        }
        std::array<SessionState, 3> states;
        std::array<SessionState, 3> references;
        std::array<SessionStatistics, 3> statistics;
        std::array<SessionStatistics, 3> reference_statistics;
        std::array<uint64_t, 3> positions = {};
        std::array<std::vector<float>, 3> logits;
        std::array<const float*, 3> output_data;
        for (size_t row = 0; row < logits.size(); ++row)
        {
            logits[row].reserve(compiled.descriptor.vocabulary_size);
            output_data[row] = logits[row].data();
        }
        std::vector<DecodeBatchEntry> entries;
        for (size_t count : {3u, 1u, 2u, 3u})
        {
            entries.clear();
            for (size_t row = 0; row < count; ++row)
            {
                const size_t index = count - row - 1;
                entries.push_back({static_cast<int32_t>((positions[index] + index) % compiled.descriptor.vocabulary_size),
                                   &statistics[index], &states[index], positions[index],
                                   row != 1 ? &logits[row] : nullptr});
            }
            auto result = forward_decode_batch(execution_model, entries, workspace);
            check(static_cast<bool>(result));
            for (size_t row = 0; row < count; ++row)
            {
                const size_t index = count - row - 1;
                const std::array<int32_t, 1> input = {entries[row].input_id};
                auto expected_status = forward_model(compiled, input, reference_statistics[index], references[index], positions[index]);
                check(static_cast<bool>(expected_status));
                check(references[index].logits.rows() == 1);
                if (entries[row].logits)
                {
                    check(logits[row].data() == output_data[row]);
                    check(logits[row].size() == references[index].logits.columns());
                    for (size_t column = 0; column < logits[row].size(); ++column)
                        check_near(logits[row][column], references[index].logits.row(0)[column], 1e-5f);
                }
                else
                    check(logits[row].empty());
                check(states[index].expert_scratch.staged_merged.allocated_bytes() == 0);
                ++positions[index];
            }
            check_released();
        }
        check(execution_model.net.layers[execution_model.net.order.back()].kind == LayerKind::LmHead);
        execution_model.net.order.pop_back();
        auto missing_logits = forward_decode_batch(execution_model, entries, workspace);
        check(!missing_logits);
        check(missing_logits.error().message == "staged execution graph did not produce logits");
        check(logits[0].empty() && logits[2].empty());
        check_released();
    };
    run_model(moe_model);
    run_model(large_model);
    const uint64_t capacity = workspace.expert.staged_merged.allocated_bytes();
    check(capacity > 0);
    run_model(attention_model);
    check(workspace.expert.staged_merged.allocated_bytes() >= capacity);
    for (bool exception : {false, true})
    {
        std::array<SessionState, 2> states;
        std::array<SessionStatistics, 2> statistics;
        std::array<std::vector<float>, 2> logits = {{{123.0f}, {456.0f}}};
        const std::array<DecodeBatchEntry, 2> entries = {{
            {0, &statistics[0], &states[0], 0, &logits[0]},
            {1, &statistics[1], &states[1], 0, &logits[1]},
        }};
        {
            ScopedLmHeadFailure failure(*attention_model, exception);
            bool failed = false;
            try
            {
                auto result = forward_decode_batch(model_compiled(*attention_model), entries, workspace);
                failed = !result;
            }
            catch (const std::exception&)
            {
                failed = true;
            }
            check(failed);
        }
        check(logits[0].empty() && logits[1].empty());
        check_released();
        run_model(attention_model);
    }
    run_model(moe_model);
}

void test_generation_logits_ownership()
{
    AttentionPackage package(false, 0, false, false);
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt).value();
    for (uint32_t chunk_size : {1u, 8u})
    {
        SessionOptions session_options;
        session_options.prefill_chunk_size = chunk_size;
        session_options.use_speculative_context = false;
        auto session = runtime.create_session(model, session_options).value();
        auto control = runtime.create_session(model, session_options).value();
        const std::array<int32_t, 3> prompt = {0, 1, 0};
        auto prefill = control->prefill(prompt);
        check(static_cast<bool>(prefill));
        const std::vector<float> retained_prefill = prefill.value().logits;
        GenerationOptions generation;
        generation.max_new_tokens = 5;
        generation.use_speculative = false;
        generation.sampling.temperature = 0.0f;
        std::vector<int32_t> expected;
        std::vector<float> logits = retained_prefill;
        std::vector<DecodeResult> retained_results;
        std::vector<std::vector<float>> saved_results;
        for (uint32_t index = 0; index < generation.max_new_tokens; ++index)
        {
            const auto sampled = control->sample(logits, generation.sampling);
            check(static_cast<bool>(sampled));
            expected.push_back(sampled.value().token_id);
            if (index + 1 == generation.max_new_tokens)
                break;
            auto decoded = control->decode(expected.back());
            check(static_cast<bool>(decoded));
            logits = decoded.value().logits;
            saved_results.push_back(logits);
            retained_results.push_back(std::move(decoded).value());
        }
        auto generated = session->generate(prompt, generation);
        check(static_cast<bool>(generated));
        check(generated.value().tokens.size() == expected.size());
        for (size_t index = 0; index < expected.size(); ++index)
            check(generated.value().tokens[index].token_id == expected[index]);
        check(session->sequence_length() == control->sequence_length());
        auto actual = session->decode(1);
        auto reference = control->decode(1);
        check(static_cast<bool>(actual));
        check(static_cast<bool>(reference));
        check(actual.value().logits == reference.value().logits);
        check(static_cast<bool>(control->reset()));
        check(prefill.value().logits == retained_prefill);
        for (size_t index = 0; index < retained_results.size(); ++index)
            check(retained_results[index].logits == saved_results[index]);
    }
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
            auto result = session.value()->generate(std::vector<int32_t>{0}, opt, [&](const StreamToken&) -> bool {
                    callback_called = true;
                    check(session.value()->metrics().timing.active);
                    throw std::runtime_error("callback failed"); }, [&](int32_t) -> std::string {
                    check(session.value()->metrics().timing.active);
                    if (throw_from_decoder)
                        throw std::runtime_error("decoder failed");
                    return "token"; });
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
    SessionMetrics callback_metrics;
    auto modified = session.value()->generate(std::vector<int32_t>{0}, opt, [&](const StreamToken& token) {
        check(static_cast<bool>(session.value()->decode(token.token_id)));
        callback_metrics = session.value()->metrics();
        check(callback_metrics.timing.active);
        check(callback_metrics.generation.prefill_tokens == 0);
        check(callback_metrics.cumulative.prefill_tokens == 1);
        check(callback_metrics.generation.decode_tokens == 0);
        check(callback_metrics.cumulative.decode_tokens == 1);
        check(callback_metrics.generation.expert_gpu_cache_resident_size
              == callback_metrics.cumulative.expert_gpu_cache_resident_size);
        check(callback_metrics.generation.expert_gpu_cache_pending_size
              == callback_metrics.cumulative.expert_gpu_cache_pending_size);
        check(callback_metrics.generation.expert_cache_resident_size
              == callback_metrics.cumulative.expert_cache_resident_size);
        check(callback_metrics.generation.kv_cache_logical_size
              == callback_metrics.cumulative.kv_cache_logical_size);
        check(callback_metrics.generation.kv_cache_allocated_size
              == callback_metrics.cumulative.kv_cache_allocated_size);
        return true;
    });
    check(!modified);
    check(modified.error().message == "generation callback modified the Session");
    const SessionMetrics modified_metrics = session.value()->metrics();
    check(!modified_metrics.timing.active);
    check(modified_metrics.generation.decode_tokens == 0);
    check(modified_metrics.cumulative.decode_tokens == 1);

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

} // namespace moe
} // namespace ncnn
