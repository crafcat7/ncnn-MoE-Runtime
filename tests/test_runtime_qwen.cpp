#include "test_runtime_decls.h"
#include "test_runtime_support.h"

namespace ncnn {
namespace moe {

void test_latent_attention_scratch_reuse()
{
    WeightStore weights;
    auto add_float = [&](std::vector<uint32_t> shape, std::vector<float> values) {
        TensorData tensor;
        tensor.dtype = DType::Float32;
        tensor.shape = std::move(shape);
        tensor.float32_data = std::move(values);
        auto added = weights.add("latent_test_" + std::to_string(weights.size()),
                                 std::move(tensor));
        check(static_cast<bool>(added));
        return added.value();
    };
    auto add_float8 = [&](std::vector<uint32_t> shape, uint32_t seed) {
        TensorData tensor;
        tensor.dtype = DType::Float8E4M3;
        tensor.shape = std::move(shape);
        const size_t element_count = tensor.element_count();
        std::shared_ptr<uint8_t[]> storage(new uint8_t[element_count], std::default_delete<uint8_t[]>());
        for (size_t index = 0; index < element_count; ++index)
        {
            const float value = static_cast<float>(static_cast<int>((index * 13 + seed * 7) % 29)
                                                   - 14)
                                * 0.0625f;
            storage[index] = float_to_float8_e4m3(value);
        }
        tensor.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
        tensor.mapped_size = element_count;
        tensor.quantization_scales.assign(1, 1.0f);
        auto added = weights.add("latent_test_float8_" + std::to_string(weights.size()),
                                 std::move(tensor));
        check(static_cast<bool>(added));
        return added.value();
    };

    AttentionBlockPlan plan;
    plan.norm_epsilon = 1e-6f;
    plan.kind = AttentionKind::MultiHeadLatent;
    plan.head_count = 1;
    plan.kv_head_count = 1;
    plan.head_dimension = 2;
    plan.sliding_window = 4;
    plan.initial_context_length = 16;
    plan.query_lora_rank = 1;
    plan.rope_head_dimension = 2;
    plan.output_lora_rank = 1;
    plan.output_group_count = 1;
    plan.pre_attention_norm_weight = add_float({2}, {1.0f, 1.0f});
    plan.query_a_weight = add_float({1, 2}, {0.5f, -0.25f});
    plan.query_norm_weight = add_float({1}, {1.0f});
    plan.query_b_weight = add_float({2, 1}, {1.0f, 0.5f});
    plan.key_value_weight = add_float({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    plan.key_value_norm_weight = add_float({2}, {1.0f, 1.0f});
    plan.sinks = add_float({1}, {0.0f});
    plan.output_b_weight = add_float({2, 1}, {1.0f, -0.25f});

    TensorData output_a;
    output_a.dtype = DType::Float8E4M3;
    output_a.shape = {1, 2};
    std::shared_ptr<uint8_t[]> output_a_storage(new uint8_t[2], std::default_delete<uint8_t[]>());
    output_a_storage[0] = float_to_float8_e4m3(1.0f);
    output_a_storage[1] = float_to_float8_e4m3(0.5f);
    output_a.mapped_data = std::shared_ptr<const uint8_t>(output_a_storage, output_a_storage.get());
    output_a.mapped_size = 2;
    output_a.quantization_scales = {1.0f};
    auto output_a_handle = weights.add("latent_test_output_a", std::move(output_a));
    check(static_cast<bool>(output_a_handle));
    plan.output_a_weight = output_a_handle.value();

    CompiledOperatorTable operators;
    operators.bind_weight_count(weights.size());
    ActivationBuffer input(2, 2);
    input.row(0)[0] = 0.25f;
    input.row(0)[1] = -0.5f;
    input.row(1)[0] = 0.75f;
    input.row(1)[1] = 0.125f;
    AttentionScratch scratch;
    LayerCache first_cache;
    LayerCache second_cache;
    ActivationBuffer first_output;
    auto first = forward_latent_attention(weights, operators, plan, ExecutionBackend::Cpu, 0,
                                          first_cache, scratch, input, first_output, g_test_optimization_flags);
    check(static_cast<bool>(first));
    check(first_cache.latent_token_count == input.rows());
    check(second_cache.latent_token_count == 0);

    const std::byte* normalized_storage = scratch.normalized.bytes().data();
    const std::byte* key_storage = scratch.key.bytes().data();
    ActivationBuffer& second_output = scratch.output;
    second_output.reset(2, 2, false);
    std::fill_n(second_output.row(0), second_output.values().size(), std::numeric_limits<float>::quiet_NaN());
    auto second = forward_latent_attention(weights, operators, plan, ExecutionBackend::Cpu, 0,
                                           second_cache, scratch, input, second_output, g_test_optimization_flags);
    check(static_cast<bool>(second));
    check(first_cache.latent_token_count == input.rows());
    check(second_cache.latent_token_count == input.rows());
    check(scratch.normalized.bytes().data() == normalized_storage);
    check(scratch.key.bytes().data() == key_storage);
    check(first_output.rows() == second_output.rows());
    check(first_output.columns() == second_output.columns());
    for (size_t index = 0; index < first_output.values().size(); ++index)
        check_near(first_output.values()[index], second_output.values()[index], 1e-6f);
    auto aliased = forward_latent_attention(weights, operators, plan, ExecutionBackend::Cpu, input.rows(),
                                            second_cache, scratch, input, input, g_test_optimization_flags);
    check(!aliased);
    check(aliased.error().code == ErrorCode::InvalidArgument);
    check(second_cache.latent_token_count == 2);
    check(input.row(0)[0] == 0.25f && input.row(1)[1] == 0.125f);

    const TensorHandle compressor_value_weight = add_float8({4, 2}, 1);
    const TensorHandle compressor_gate_weight = add_float8({4, 2}, 2);
    const TensorHandle mixed_compressor_gate_weight = add_float({4, 2},
                                                                {0.125f, -0.25f, 0.375f, -0.5f, 0.625f, -0.75f, 0.875f, -1.0f});
    const TensorHandle compressor_position = add_float({4, 4},
                                                       std::vector<float>(16, 0.0f));
    const TensorHandle compressor_norm = add_float({2}, {1.0f, 1.0f});
    AttentionBlockPlan compressed_plan = plan;
    compressed_plan.compression_ratio = 4;
    compressed_plan.index_top_k = 1;
    compressed_plan.index_head_count = 1;
    compressed_plan.index_head_dimension = 2;
    compressed_plan.compressor_key_value_weight = compressor_value_weight;
    compressed_plan.compressor_gate_weight = compressor_gate_weight;
    compressed_plan.compressor_position = compressor_position;
    compressed_plan.compressor_norm_weight = compressor_norm;
    compressed_plan.indexer_compressor_key_value_weight = compressor_value_weight;
    compressed_plan.indexer_compressor_gate_weight = compressor_gate_weight;
    compressed_plan.indexer_compressor_position = compressor_position;
    compressed_plan.indexer_compressor_norm_weight = compressor_norm;
    compressed_plan.indexer_query_weight = plan.query_b_weight;
    compressed_plan.indexer_weights_weight = plan.query_a_weight;
    AttentionBlockPlan mixed_plan = compressed_plan;
    mixed_plan.compressor_gate_weight = mixed_compressor_gate_weight;
    operators.bind_weight_count(weights.size());

    ActivationBuffer compressed_input(4, 2);
    for (size_t row = 0; row < compressed_input.rows(); ++row)
    {
        compressed_input.row(row)[0] = 0.125f + static_cast<float>(row) * 0.25f;
        compressed_input.row(row)[1] = -0.375f + static_cast<float>(row) * 0.125f;
    }
    const TensorData& float8_compressor_value = weights.at(compressor_value_weight);
    const TensorData& float8_compressor_gate = weights.at(compressor_gate_weight);
    const TensorData& mixed_compressor_gate = weights.at(mixed_compressor_gate_weight);
    ActivationBuffer pair_probe_values;
    ActivationBuffer pair_probe_scores;
    ActivationBuffer pair_probe_scratch;
    check(forward_linear({&float8_compressor_value, &float8_compressor_gate},
                         compressed_input,
                         {&pair_probe_values, &pair_probe_scores},
                         g_test_optimization_flags,
                         {},
                         &pair_probe_scratch));
    ActivationBuffer mixed_probe_values;
    ActivationBuffer mixed_probe_scores;
    check(!forward_linear({&float8_compressor_value, &mixed_compressor_gate},
                          compressed_input,
                          {&mixed_probe_values, &mixed_probe_scores},
                          g_test_optimization_flags));

    auto check_activation_parity = [&](const ActivationBuffer& left, const ActivationBuffer& right) {
        check(left.rows() == right.rows());
        check(left.columns() == right.columns());
        for (size_t index = 0; index < left.values().size(); ++index)
            check_near(left.values()[index], right.values()[index], 1e-6f);
    };
    auto check_cache_parity = [&](const LayerCache& left, const LayerCache& right) {
        check(left.latent_token_count == right.latent_token_count);
        check(left.capacity_tokens == right.capacity_tokens);
        check(left.columns == right.columns);
        check(left.latent_window == right.latent_window);
        check(left.latent_compressed == right.latent_compressed);
        check(left.latent_index_compressed == right.latent_index_compressed);
        check(left.compressor_pending_values == right.compressor_pending_values);
        check(left.compressor_pending_scores == right.compressor_pending_scores);
        check(left.compressor_previous_values == right.compressor_previous_values);
        check(left.compressor_previous_scores == right.compressor_previous_scores);
        check(left.index_compressor_pending_values == right.index_compressor_pending_values);
        check(left.index_compressor_pending_scores == right.index_compressor_pending_scores);
        check(left.index_compressor_previous_values == right.index_compressor_previous_values);
        check(left.index_compressor_previous_scores == right.index_compressor_previous_scores);
        check(left.latent_selected_indices == right.latent_selected_indices);
        check(left.latent_attention_logits == right.latent_attention_logits);
        check(left.latent_rope_cosines == right.latent_rope_cosines);
        check(left.latent_rope_sines == right.latent_rope_sines);
    };
    auto poison_buffer = [](ActivationBuffer& buffer) {
        const float poison = std::numeric_limits<float>::quiet_NaN();
        for (size_t offset = 0; offset < buffer.bytes().size(); offset += sizeof(float))
            std::memcpy(buffer.mutable_bytes().data() + offset, &poison, sizeof(poison));
    };
    auto poison_latent_scratch = [&](AttentionScratch& scenario_scratch) {
        poison_buffer(scenario_scratch.quantized_input);
        poison_buffer(scenario_scratch.latent_compressor_values);
        poison_buffer(scenario_scratch.latent_compressor_scores);
        poison_buffer(scenario_scratch.latent_index_compressor_values);
        poison_buffer(scenario_scratch.latent_index_compressor_scores);
        poison_buffer(scenario_scratch.latent_token_input);
        poison_buffer(scenario_scratch.latent_token_rank);
        poison_buffer(scenario_scratch.latent_index_query);
        poison_buffer(scenario_scratch.latent_index_projected_weights);
        const float poison = std::numeric_limits<float>::quiet_NaN();
        std::fill(scenario_scratch.latent_compressor_pooled.begin(),
                  scenario_scratch.latent_compressor_pooled.end(), poison);
        std::fill(scenario_scratch.latent_compressor_exponentials.begin(),
                  scenario_scratch.latent_compressor_exponentials.end(), poison);
        std::fill(scenario_scratch.index_scores.begin(),
                  scenario_scratch.index_scores.end(),
                  std::pair<float, uint32_t>{poison,
                                             std::numeric_limits<uint32_t>::max()});
    };
    auto run_latent = [&](const AttentionBlockPlan& scenario_plan,
                          uint64_t position,
                          const ActivationBuffer& scenario_input,
                          LayerCache& cache,
                          AttentionScratch& scenario_scratch,
                          ActivationBuffer& output) {
        auto result = forward_latent_attention(weights,
                                               operators,
                                               scenario_plan,
                                               ExecutionBackend::Cpu,
                                               position,
                                               cache,
                                               scenario_scratch,
                                               scenario_input,
                                               output,
                                               g_test_optimization_flags);
        check(static_cast<bool>(result));
    };

    ActivationBuffer r4_full_input(8, 2);
    for (size_t row = 0; row < r4_full_input.rows(); ++row)
    {
        r4_full_input.row(row)[0] = 0.0625f + static_cast<float>((row * 3) % 13) * 0.03125f;
        r4_full_input.row(row)[1] = -0.3125f + static_cast<float>((row * 5) % 11) * 0.0625f;
    }
    LayerCache r4_batch_cache;
    LayerCache r4_serial_cache;
    AttentionScratch r4_batch_scratch;
    AttentionScratch r4_serial_scratch;
    ActivationBuffer r4_batch_output;
    ActivationBuffer r4_serial_output(8, 2);
    run_latent(compressed_plan, 0, r4_full_input, r4_batch_cache,
               r4_batch_scratch, r4_batch_output);
    for (size_t row = 0; row < r4_full_input.rows(); ++row)
    {
        ActivationBuffer token(1, 2);
        std::copy_n(r4_full_input.row(row), 2, token.row(0));
        ActivationBuffer token_output;
        run_latent(compressed_plan, row, token, r4_serial_cache,
                   r4_serial_scratch, token_output);
        std::copy_n(token_output.row(0), 2, r4_serial_output.row(row));
    }
    check_activation_parity(r4_batch_output, r4_serial_output);
    check_cache_parity(r4_batch_cache, r4_serial_cache);

    LayerCache pair_cache;
    LayerCache pair_reference_cache;
    AttentionScratch pair_scratch;
    AttentionScratch pair_reference_scratch;
    ActivationBuffer pair_output;
    ActivationBuffer pair_reference_output;
    run_latent(compressed_plan, 0, compressed_input, pair_cache, pair_scratch, pair_output);
    run_latent(compressed_plan,
               0,
               compressed_input,
               pair_reference_cache,
               pair_reference_scratch,
               pair_reference_output);
    check(pair_cache.latent_compressed.size() == 2);
    check(pair_cache.latent_index_compressed.size() == 2);
    check(pair_scratch.quantized_input.rows() == compressed_input.rows());
    const uint64_t pair_quantized_capacity = pair_scratch.quantized_input.allocated_bytes();
    const std::byte* pair_quantized_address = pair_scratch.quantized_input.bytes().data();
    check_activation_parity(pair_output, pair_reference_output);
    check_cache_parity(pair_cache, pair_reference_cache);
    poison_latent_scratch(pair_scratch);
    AttentionScratch pair_second_reference_scratch;
    ActivationBuffer pair_second_output;
    ActivationBuffer pair_second_reference_output;
    run_latent(compressed_plan, 4, compressed_input, pair_cache, pair_scratch, pair_second_output);
    run_latent(compressed_plan,
               4,
               compressed_input,
               pair_reference_cache,
               pair_second_reference_scratch,
               pair_second_reference_output);
    check(pair_scratch.quantized_input.allocated_bytes() == pair_quantized_capacity);
    check(pair_scratch.quantized_input.bytes().data() == pair_quantized_address);
    check_activation_parity(pair_second_output, pair_second_reference_output);
    check_cache_parity(pair_cache, pair_reference_cache);
    check(pair_cache.latent_selected_indices.size() == 1);
    check(pair_scratch.latent_index_query.rows() == 1);
    check(pair_scratch.latent_index_projected_weights.rows() == 1);
    const std::vector<uint32_t> pair_retained_indices = pair_cache.latent_selected_indices;
    const std::vector<float> pair_retained_logits = pair_cache.latent_attention_logits;

    AttentionBlockPlan wide_plan = compressed_plan;
    wide_plan.compression_ratio = 128;
    wide_plan.compressor_key_value_weight = add_float8({2, 2}, 3);
    wide_plan.compressor_gate_weight = add_float8({2, 2}, 4);
    wide_plan.compressor_position = add_float({128, 2}, std::vector<float>(256, 0.0f));
    operators.bind_weight_count(weights.size());
    ActivationBuffer wide_input(128, 2);
    for (size_t row = 0; row < wide_input.rows(); ++row)
    {
        wide_input.row(row)[0] = 0.0625f + static_cast<float>(row % 11) * 0.03125f;
        wide_input.row(row)[1] = -0.25f + static_cast<float>(row % 7) * 0.0625f;
    }
    ActivationBuffer wide_prefix(127, 2);
    for (size_t row = 0; row < wide_prefix.rows(); ++row)
        std::copy_n(wide_input.row(row), 2, wide_prefix.row(row));
    ActivationBuffer wide_tail(1, 2);
    std::copy_n(wide_input.row(127), 2, wide_tail.row(0));
    LayerCache wide_cache;
    LayerCache wide_reference_cache;
    LayerCache wide_serial_cache;
    AttentionScratch wide_reference_scratch;
    AttentionScratch wide_serial_scratch;
    ActivationBuffer wide_prefix_output;
    ActivationBuffer wide_prefix_reference_output;
    ActivationBuffer wide_serial_output(128, 2);
    poison_latent_scratch(pair_scratch);
    run_latent(wide_plan, 0, wide_prefix, wide_cache, pair_scratch, wide_prefix_output);
    run_latent(wide_plan, 0, wide_prefix, wide_reference_cache,
               wide_reference_scratch, wide_prefix_reference_output);
    for (size_t row = 0; row < wide_prefix.rows(); ++row)
    {
        ActivationBuffer token(1, 2);
        std::copy_n(wide_input.row(row), 2, token.row(0));
        ActivationBuffer token_output;
        run_latent(wide_plan, row, token, wide_serial_cache,
                   wide_serial_scratch, token_output);
        std::copy_n(token_output.row(0), 2, wide_serial_output.row(row));
    }
    check_activation_parity(wide_prefix_output, wide_prefix_reference_output);
    check_cache_parity(wide_cache, wide_reference_cache);
    for (size_t row = 0; row < wide_prefix.rows(); ++row)
        for (size_t column = 0; column < wide_prefix_output.columns(); ++column)
            check_near(wide_prefix_output.row(row)[column], wide_serial_output.row(row)[column], 1e-6f);
    check_cache_parity(wide_cache, wide_serial_cache);
    check(wide_cache.latent_compressed.empty());
    check(wide_cache.compressor_pending_values.size() == 128 * wide_plan.head_dimension);
    poison_latent_scratch(pair_scratch);
    ActivationBuffer wide_tail_output;
    ActivationBuffer wide_tail_reference_output;
    run_latent(wide_plan, 127, wide_tail, wide_cache, pair_scratch, wide_tail_output);
    run_latent(wide_plan, 127, wide_tail, wide_reference_cache,
               wide_reference_scratch, wide_tail_reference_output);
    ActivationBuffer wide_serial_tail_output;
    run_latent(wide_plan, 127, wide_tail, wide_serial_cache,
               wide_serial_scratch, wide_serial_tail_output);
    check_activation_parity(wide_tail_output, wide_tail_reference_output);
    check_cache_parity(wide_cache, wide_reference_cache);
    check_activation_parity(wide_tail_output, wide_serial_tail_output);
    check_cache_parity(wide_cache, wide_serial_cache);
    std::copy_n(wide_serial_tail_output.row(0), 2, wide_serial_output.row(127));
    LayerCache wide_full_cache;
    AttentionScratch wide_full_scratch;
    ActivationBuffer wide_full_output;
    run_latent(wide_plan, 0, wide_input, wide_full_cache,
               wide_full_scratch, wide_full_output);
    check_activation_parity(wide_full_output, wide_serial_output);
    check_cache_parity(wide_full_cache, wide_serial_cache);
    check(wide_cache.latent_compressed.size() == wide_plan.head_dimension);
    const size_t wide_exponential_capacity = pair_scratch.latent_compressor_exponentials.capacity();
    const uint64_t wide_key_capacity = pair_scratch.key.allocated_bytes();
    const std::byte* const wide_key_address = pair_scratch.key.bytes().data();
    check(pair_cache.latent_selected_indices == pair_retained_indices);
    check(pair_cache.latent_attention_logits == pair_retained_logits);

    poison_latent_scratch(pair_scratch);
    ActivationBuffer pair_third_output;
    ActivationBuffer pair_third_reference_output;
    run_latent(compressed_plan, 8, compressed_input, pair_cache, pair_scratch, pair_third_output);
    run_latent(compressed_plan, 8, compressed_input,
               pair_reference_cache, pair_reference_scratch, pair_third_reference_output);
    check_activation_parity(pair_third_output, pair_third_reference_output);
    check_cache_parity(pair_cache, pair_reference_cache);
    check(pair_scratch.latent_compressor_exponentials.size() == 8);
    check(pair_scratch.latent_compressor_exponentials.capacity() == wide_exponential_capacity);
    check(pair_scratch.key.allocated_bytes() == wide_key_capacity);
    check(pair_scratch.key.bytes().data() == wide_key_address);
    const uint64_t pair_index_score_capacity = pair_scratch.index_scores.capacity();
    const std::pair<float, uint32_t>* const pair_index_score_data = pair_scratch.index_scores.data();
    const std::vector<uint32_t> latest_pair_indices = pair_cache.latent_selected_indices;
    const std::vector<float> latest_pair_logits = pair_cache.latent_attention_logits;

    LayerCache mixed_cache;
    LayerCache mixed_reference_cache;
    AttentionScratch mixed_reference_scratch;
    ActivationBuffer mixed_output;
    ActivationBuffer mixed_reference_output;
    poison_latent_scratch(pair_scratch);
    run_latent(mixed_plan, 0, compressed_input, mixed_cache, pair_scratch, mixed_output);
    run_latent(mixed_plan,
               0,
               compressed_input,
               mixed_reference_cache,
               mixed_reference_scratch,
               mixed_reference_output);
    check(mixed_cache.latent_compressed.size() == 2);
    check(mixed_cache.latent_index_compressed.size() == 2);
    check(pair_scratch.quantized_input.rows() == compressed_input.rows());
    check_activation_parity(mixed_output, mixed_reference_output);
    check_cache_parity(mixed_cache, mixed_reference_cache);
    poison_latent_scratch(pair_scratch);
    AttentionScratch mixed_second_reference_scratch;
    ActivationBuffer mixed_second_output;
    ActivationBuffer mixed_second_reference_output;
    run_latent(mixed_plan, 4, compressed_input, mixed_cache, pair_scratch, mixed_second_output);
    run_latent(mixed_plan,
               4,
               compressed_input,
               mixed_reference_cache,
               mixed_second_reference_scratch,
               mixed_second_reference_output);
    check_activation_parity(mixed_second_output, mixed_second_reference_output);
    check_cache_parity(mixed_cache, mixed_reference_cache);
    check(mixed_cache.latent_selected_indices.size() == 1);
    check(pair_scratch.latent_index_query.rows() == 1);
    check(pair_scratch.latent_index_projected_weights.rows() == 1);
    check(pair_scratch.index_scores.capacity() == pair_index_score_capacity);
    check(pair_scratch.index_scores.data() == pair_index_score_data);
    check(pair_cache.latent_selected_indices == latest_pair_indices);
    check(pair_cache.latent_attention_logits == latest_pair_logits);

    ActivationBuffer alternate_compressed_input(4, 2);
    for (size_t row = 0; row < alternate_compressed_input.rows(); ++row)
    {
        alternate_compressed_input.row(row)[0] = -0.25f + static_cast<float>(row) * 0.125f;
        alternate_compressed_input.row(row)[1] = 0.5f - static_cast<float>(row) * 0.1875f;
    }
    LayerCache batch_first_cache;
    LayerCache batch_second_cache;
    LayerCache batch_first_reference_cache;
    LayerCache batch_second_reference_cache;
    ActivationBuffer ignored_output;
    auto seed_batch_cache = [&](LayerCache& cache, const ActivationBuffer& seed_input) {
        run_latent(compressed_plan, 0, seed_input, cache, pair_scratch, ignored_output);
        run_latent(compressed_plan, 4, seed_input, cache, pair_scratch, ignored_output);
    };
    seed_batch_cache(batch_first_cache, compressed_input);
    seed_batch_cache(batch_second_cache, alternate_compressed_input);
    seed_batch_cache(batch_first_reference_cache, compressed_input);
    seed_batch_cache(batch_second_reference_cache, alternate_compressed_input);
    ActivationBuffer batch_input(2, 2);
    std::copy_n(compressed_input.row(0), 2, batch_input.row(0));
    std::copy_n(alternate_compressed_input.row(0), 2, batch_input.row(1));
    const std::array<uint64_t, 2> batch_positions{8, 8};
    const std::array<LayerCache*, 2> batch_caches{&batch_first_cache, &batch_second_cache};
    ActivationBuffer batch_output;
    auto batch_result = forward_latent_batch(weights, operators, compressed_plan,
                                             ExecutionBackend::Cpu, batch_positions,
                                             batch_caches, pair_scratch, batch_input,
                                             batch_output, g_test_optimization_flags);
    check(static_cast<bool>(batch_result));
    ActivationBuffer first_batch_input(1, 2);
    ActivationBuffer second_batch_input(1, 2);
    std::copy_n(batch_input.row(0), 2, first_batch_input.row(0));
    std::copy_n(batch_input.row(1), 2, second_batch_input.row(0));
    ActivationBuffer first_batch_reference_output;
    ActivationBuffer second_batch_reference_output;
    run_latent(compressed_plan, 8, first_batch_input, batch_first_reference_cache,
               pair_scratch, first_batch_reference_output);
    run_latent(compressed_plan, 8, second_batch_input, batch_second_reference_cache,
               pair_scratch, second_batch_reference_output);
    check(batch_output.rows() == batch_positions.size());
    check(batch_output.columns() == first_batch_reference_output.columns());
    for (uint32_t column = 0; column < batch_output.columns(); ++column)
    {
        check_near(batch_output.row(0)[column], first_batch_reference_output.row(0)[column], 1e-6f);
        check_near(batch_output.row(1)[column], second_batch_reference_output.row(0)[column], 1e-6f);
    }
    check_cache_parity(batch_first_cache, batch_first_reference_cache);
    check_cache_parity(batch_second_cache, batch_second_reference_cache);
    check(pair_cache.latent_selected_indices == latest_pair_indices);
    check(pair_cache.latent_attention_logits == latest_pair_logits);

    AttentionBlockPlan dspark_plan = plan;
    dspark_plan.compression_ratio = 0;
    ActivationBuffer dspark_input(3, 2);
    for (size_t row = 0; row < dspark_input.rows(); ++row)
    {
        dspark_input.row(row)[0] = 0.25f + static_cast<float>(row) * 0.125f;
        dspark_input.row(row)[1] = -0.5f + static_cast<float>(row) * 0.25f;
    }
    auto append_dspark = [&](LayerCache& cache,
                             AttentionScratch& scenario_scratch,
                             uint64_t position,
                             const ActivationBuffer& scenario_input) {
        auto result = append_dspark_attention_context(weights, operators, dspark_plan,
                                                      ExecutionBackend::Cpu, position, cache,
                                                      scenario_scratch, scenario_input,
                                                      g_test_optimization_flags);
        check(static_cast<bool>(result));
    };
    LayerCache dspark_cache;
    LayerCache dspark_reference_cache;
    AttentionScratch dspark_reference_scratch;
    append_dspark(dspark_cache, pair_scratch, 0, dspark_input);
    append_dspark(dspark_reference_cache, dspark_reference_scratch, 0, dspark_input);
    check_cache_parity(dspark_cache, dspark_reference_cache);
    const std::vector<float> retained_dspark_window = dspark_cache.latent_window;

    ActivationBuffer other_dspark_input(3, 2);
    for (size_t row = 0; row < other_dspark_input.rows(); ++row)
    {
        other_dspark_input.row(row)[0] = -0.375f + static_cast<float>(row) * 0.0625f;
        other_dspark_input.row(row)[1] = 0.75f - static_cast<float>(row) * 0.125f;
    }
    LayerCache other_dspark_cache;
    LayerCache other_dspark_reference_cache;
    AttentionScratch other_dspark_reference_scratch;
    poison_buffer(pair_scratch.key);
    poison_buffer(pair_scratch.quantized_input);
    append_dspark(other_dspark_cache, pair_scratch, 0, other_dspark_input);
    append_dspark(other_dspark_reference_cache, other_dspark_reference_scratch,
                  0, other_dspark_input);
    check_cache_parity(other_dspark_cache, other_dspark_reference_cache);
    check(dspark_cache.latent_window == retained_dspark_window);

    ActivationBuffer dspark_tail(2, 2);
    std::copy_n(dspark_input.row(1), 2, dspark_tail.row(0));
    std::copy_n(dspark_input.row(2), 2, dspark_tail.row(1));
    poison_buffer(pair_scratch.key);
    poison_buffer(pair_scratch.quantized_input);
    append_dspark(dspark_cache, pair_scratch, 3, dspark_tail);
    append_dspark(dspark_reference_cache, dspark_reference_scratch, 3, dspark_tail);
    check_cache_parity(dspark_cache, dspark_reference_cache);
    check(dspark_cache.latent_token_count == 5);
    const LayerCache before_alias_failure = dspark_cache;
    const std::vector<std::byte> key_before_alias_failure(pair_scratch.key.bytes().begin(),
                                                          pair_scratch.key.bytes().end());
    const std::byte* const key_address_before_alias_failure = pair_scratch.key.bytes().data();
    const uint64_t key_capacity_before_alias_failure = pair_scratch.key.allocated_bytes();
    const std::vector<std::byte> quantized_before_alias_failure(pair_scratch.quantized_input.bytes().begin(),
                                                                pair_scratch.quantized_input.bytes().end());
    auto alias_failure = append_dspark_attention_context(weights, operators, dspark_plan,
                                                         ExecutionBackend::Cpu, 5, dspark_cache,
                                                         pair_scratch, pair_scratch.key,
                                                         g_test_optimization_flags);
    check(!alias_failure);
    check(alias_failure.error().code == ErrorCode::InvalidArgument);
    check_cache_parity(dspark_cache, before_alias_failure);
    check(pair_scratch.key.bytes().data() == key_address_before_alias_failure);
    check(pair_scratch.key.allocated_bytes() == key_capacity_before_alias_failure);
    check(std::equal(pair_scratch.key.bytes().begin(), pair_scratch.key.bytes().end(),
                     key_before_alias_failure.begin(), key_before_alias_failure.end()));
    check(std::equal(pair_scratch.quantized_input.bytes().begin(),
                     pair_scratch.quantized_input.bytes().end(),
                     quantized_before_alias_failure.begin(),
                     quantized_before_alias_failure.end()));

    dspark_cache = LayerCache{};
    dspark_reference_cache = LayerCache{};
    ActivationBuffer reset_dspark_input(1, 2);
    std::copy_n(dspark_input.row(0), 2, reset_dspark_input.row(0));
    append_dspark(dspark_cache, pair_scratch, 17, reset_dspark_input);
    append_dspark(dspark_reference_cache, dspark_reference_scratch,
                  17, reset_dspark_input);
    check_cache_parity(dspark_cache, dspark_reference_cache);
    check(dspark_cache.latent_token_count == 18);

    CompiledModel dspark_model;
    dspark_model.descriptor.hidden_size = 2;
    dspark_model.descriptor.norm_epsilon = dspark_plan.norm_epsilon;
    dspark_model.descriptor.norm_weight_offset = dspark_plan.norm_weight_offset;
    dspark_model.weights = weights;
    dspark_model.operators.bind_weight_count(dspark_model.weights.size());
    dspark_model.speculative.kind = SpeculativeModelKind::DSpark;
    dspark_model.speculative.block_size = 1;
    dspark_model.speculative.target_layer_ids = {0};
    dspark_model.speculative.main_projection_weight = dspark_plan.key_value_weight;
    dspark_model.speculative.main_norm_weight = dspark_plan.key_value_norm_weight;
    dspark_model.speculative.net.layer_plans.resize(2);
    dspark_model.speculative.net.layers.resize(2);
    dspark_model.speculative.layer_nodes.resize(2);
    for (size_t layer = 0; layer < 2; ++layer)
    {
        dspark_model.speculative.net.layer_plans[layer].attention = dspark_plan;
        Layer& node = dspark_model.speculative.net.layers[layer];
        node.id = static_cast<LayerId>(layer);
        node.kind = LayerKind::Attention;
        node.backend = ExecutionBackend::Cpu;
        node.plan_index = static_cast<uint32_t>(layer);
        dspark_model.speculative.layer_nodes[layer].attention = static_cast<LayerId>(layer);
    }
    SessionState reused_speculative_state;
    SessionState reference_speculative_state;
    SessionStatistics reused_speculative_statistics;
    SessionStatistics reference_speculative_statistics;
    reused_speculative_state.speculative_main_hidden.reset(2, 2, false);
    for (size_t row = 0; row < 2; ++row)
        std::copy_n(dspark_input.row(row), 2, reused_speculative_state.speculative_main_hidden.row(row));
    reference_speculative_state.speculative_main_hidden = reused_speculative_state.speculative_main_hidden;
    ActivationBuffer first_speculative_source = reused_speculative_state.speculative_main_hidden;
    auto reused_update = update_speculative_context(dspark_model,
                                                    reused_speculative_statistics,
                                                    reused_speculative_state);
    auto reference_update = update_speculative_context(dspark_model,
                                                       reference_speculative_statistics,
                                                       reference_speculative_state);
    check(static_cast<bool>(reused_update));
    check(static_cast<bool>(reference_update));
    check_cache_parity(reused_speculative_state.speculative_layers[0],
                       reference_speculative_state.speculative_layers[0]);
    check_cache_parity(reused_speculative_state.speculative_layers[1],
                       reference_speculative_state.speculative_layers[1]);
    check_activation_parity(reused_speculative_state.speculative_main_hidden,
                            first_speculative_source);
    const uint64_t normalized_capacity = reused_speculative_state.attention_scratch.normalized.allocated_bytes();
    const std::byte* const normalized_address = reused_speculative_state.attention_scratch.normalized.bytes().data();
    const uint64_t key_capacity = reused_speculative_state.attention_scratch.key.allocated_bytes();
    const std::byte* const key_address = reused_speculative_state.attention_scratch.key.bytes().data();

    reused_speculative_state.speculative_main_hidden.reset(1, 2, false);
    std::copy_n(dspark_input.row(2), 2, reused_speculative_state.speculative_main_hidden.row(0));
    reused_speculative_state.speculative_main_hidden_position = 2;
    reference_speculative_state.speculative_main_hidden = reused_speculative_state.speculative_main_hidden;
    reference_speculative_state.speculative_main_hidden_position = 2;
    reference_speculative_state.attention_scratch = AttentionScratch{};
    ActivationBuffer second_speculative_source = reused_speculative_state.speculative_main_hidden;
    poison_buffer(reused_speculative_state.attention_scratch.normalized);
    poison_buffer(reused_speculative_state.attention_scratch.key);
    reused_update = update_speculative_context(dspark_model,
                                               reused_speculative_statistics,
                                               reused_speculative_state);
    reference_update = update_speculative_context(dspark_model,
                                                  reference_speculative_statistics,
                                                  reference_speculative_state);
    check(static_cast<bool>(reused_update));
    check(static_cast<bool>(reference_update));
    check_activation_parity(reused_speculative_state.speculative_main_hidden,
                            second_speculative_source);
    check(reused_speculative_state.attention_scratch.normalized.allocated_bytes()
          == normalized_capacity);
    check(reused_speculative_state.attention_scratch.normalized.bytes().data()
          == normalized_address);
    check(reused_speculative_state.attention_scratch.key.allocated_bytes() == key_capacity);
    check(reused_speculative_state.attention_scratch.key.bytes().data() == key_address);
    for (size_t layer = 0; layer < 2; ++layer)
        check_cache_parity(reused_speculative_state.speculative_layers[layer],
                           reference_speculative_state.speculative_layers[layer]);
    check(pair_cache.latent_selected_indices == latest_pair_indices);
    check(pair_cache.latent_attention_logits == latest_pair_logits);
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

static void write_qwen_mxfp4_test_artifact(ModelPackage& package,
                                           const std::filesystem::path& root)
{
    constexpr uint32_t layer_count = 4;
    constexpr uint32_t expert_count = 4;
    constexpr uint32_t hidden_size = 32;
    constexpr uint32_t intermediate_size = 32;
    package.root = root;
    package.manifest.raw_json = std::regex_replace(package.manifest.raw_json,
                                                   std::regex(R"("hidden_size"\s*:\s*16)"),
                                                   R"("hidden_size": 32)");
    package.manifest.raw_json = std::regex_replace(package.manifest.raw_json,
                                                   std::regex(R"("moe_intermediate_size"\s*:\s*4)"),
                                                   R"("moe_intermediate_size": 32)");
    package.manifest.raw_json = std::regex_replace(package.manifest.raw_json,
                                                   std::regex(R"("shared_expert_intermediate_size"\s*:\s*4)"),
                                                   R"("shared_expert_intermediate_size": 32)");
    const std::string index_json = "{}\n";
    {
        std::ofstream config(root / "config.json", std::ios::binary);
        config << package.manifest.raw_json;
    }
    {
        std::ofstream index(root / "model.safetensors.index.json",
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
        add_tensor(prefix + "gate_up.blocks",
                   {expert_count, intermediate_size * 2, hidden_size / 32, 16});
        add_tensor(prefix + "gate_up.scales",
                   {expert_count, intermediate_size * 2, hidden_size / 32});
        add_tensor(prefix + "down.blocks",
                   {expert_count, hidden_size, intermediate_size / 32, 16});
        add_tensor(prefix + "down.scales",
                   {expert_count, hidden_size, intermediate_size / 32});
    }
    const std::string mtp_prefix = "__ncnn_moe_qwen3_6_mxfp4__.mtp.layers.0.experts.";
    add_tensor(mtp_prefix + "gate_up.blocks",
               {expert_count, intermediate_size * 2, hidden_size / 32, 16});
    add_tensor(mtp_prefix + "gate_up.scales",
               {expert_count, intermediate_size * 2, hidden_size / 32});
    add_tensor(mtp_prefix + "down.blocks",
               {expert_count, hidden_size, intermediate_size / 32, 16});
    add_tensor(mtp_prefix + "down.scales",
               {expert_count, hidden_size, intermediate_size / 32});
    header << '}';
    std::string encoded_header = header.str();
    encoded_header.append((8 - encoded_header.size() % 8) % 8,
                          ' ');
    std::ofstream artifact(root / "ncnn-moe-qwen3.6-mxfp4.safetensors",
                           std::ios::binary);
    const uint64_t header_bytes = encoded_header.size();
    artifact.write(reinterpret_cast<const char*>(&header_bytes),
                   sizeof(header_bytes));
    artifact.write(encoded_header.data(),
                   static_cast<std::streamsize>(encoded_header.size()));
    std::array<char, 4096> zeros = {};
    for (uint64_t written = 0; written < data_offset;)
    {
        const uint64_t count = std::min<uint64_t>(zeros.size(),
                                                  data_offset - written);
        artifact.write(zeros.data(),
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
    const MoeDescriptor& moe = descriptor.layers[0].moe;
    check(moe.expert_weight_dtype == DType::BFloat16);
    check(moe.layout == ExpertLayout::PackedGateUpDown);
    check(moe.normalization == RouterNormalization::SelectedExperts);
    check(moe.shared_expert_count == 1);
    check(has_flag(moe.flags, MoeDescriptorSharedExpertGate));
    Option options;
    auto memory = plan_model_memory(descriptor,
                                    options,
                                    UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(memory));
    check(memory.value().selected_mode == ExpertMemoryMode::Eager);
    check(memory.value().expert_pair_size == 384);
    check(memory.value().estimated_expert_size == 6144);
    check(!has_flag(moe.flags, MoeDescriptorFileBackedExperts));
    Option unsupported_on_demand;
    unsupported_on_demand.expert_memory_mode = ExpertMemoryMode::OnDemand;
    check(!plan_model_memory(descriptor,
                             unsupported_on_demand,
                             UINT64_C(8) * 1024 * 1024 * 1024));

    ModelPackage nested_text_config = qwen3_5_moe_package();
    nested_text_config.manifest.raw_json = std::regex_replace(nested_text_config.manifest.raw_json,
                                                              std::regex(R"("rope_theta"\s*:\s*10000\.0,\s*"partial_rotary_factor"\s*:\s*0\.5)"),
                                                              R"("rope_parameters":{"rope_theta":12000.0,"partial_rotary_factor":0.5},"rope_theta":10000.0,"partial_rotary_factor":0.5)");
    nested_text_config.manifest.raw_json = std::string(R"({"text_config":)")
                                           + nested_text_config.manifest.raw_json
                                           + R"(,"hidden_size":1,"rope_theta":1.0})";
    auto nested_text_config_result = adapter.parse_model(nested_text_config);
    check(static_cast<bool>(nested_text_config_result));
    check(nested_text_config_result.value().hidden_size == 16);
    check_near(nested_text_config_result.value().layers.back().attention.rope_theta,
               12000.0f,
               1e-6f);

    ModelPackage invalid_text_config = qwen3_5_moe_package();
    invalid_text_config.manifest.raw_json = R"({"text_config":[1]})";
    auto invalid_text_config_result = adapter.parse_model(invalid_text_config);
    check(!invalid_text_config_result);
    check(invalid_text_config_result.error().message
          == "invalid Qwen3 MoE object field: text_config");

    ModelPackage invalid_rope_parameters = nested_text_config;
    invalid_rope_parameters.manifest.raw_json = std::regex_replace(invalid_rope_parameters.manifest.raw_json,
                                                                   std::regex(R"("rope_parameters"\s*:\s*\{[^{}]*\})"),
                                                                   R"("rope_parameters":[1])");
    auto invalid_rope_parameters_result = adapter.parse_model(invalid_rope_parameters);
    check(!invalid_rope_parameters_result);
    check(invalid_rope_parameters_result.error().message
          == "invalid Qwen3 MoE object field: rope_parameters");

    ModelPackage invalid = qwen3_5_moe_package();
    invalid.manifest.raw_json = std::regex_replace(invalid.manifest.raw_json,
                                                   std::regex(R"("mamba_ssm_dtype"\s*:\s*"float32")"),
                                                   R"("mamba_ssm_dtype": "bfloat16")");
    check(!adapter.parse_model(invalid));

    invalid = qwen3_5_moe_package();
    invalid.manifest.raw_json = std::regex_replace(invalid.manifest.raw_json,
                                                   std::regex(R"("num_key_value_heads"\s*:\s*1)"),
                                                   R"("num_key_value_heads": 0)");
    auto zero_kv_heads = adapter.parse_model(invalid);
    check(!zero_kv_heads);
    check(zero_kv_heads.error().code == ErrorCode::InvalidModel);
    check(zero_kv_heads.error().message
          == "unsupported Qwen3 MoE architectural dimensions");

    invalid = qwen3_5_moe_package();
    invalid.manifest.raw_json = std::regex_replace(invalid.manifest.raw_json,
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
        invalid_rotary.manifest.raw_json = std::regex_replace(invalid_rotary.manifest.raw_json,
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
    write_qwen_mxfp4_test_artifact(artifact_package,
                                   artifact_directory.path());
    auto artifact_descriptor = adapter.parse_model(artifact_package);
    check(static_cast<bool>(artifact_descriptor));
    check(artifact_descriptor.value().layers[0].moe.expert_weight_dtype
          == DType::MxFp4);
    check(artifact_descriptor.value().speculative_kind
          == SpeculativeModelKind::Mtp);
    check(artifact_descriptor.value().speculative_layer_count == 1);
    check(artifact_descriptor.value().speculative_block_size == 2);

    ModelPackage changed_manifest = artifact_package;
    changed_manifest.manifest.raw_json += ' ';
    auto changed_manifest_artifact = adapter.parse_model(changed_manifest);
    check(!changed_manifest_artifact);
    check(changed_manifest_artifact.error().code == ErrorCode::InvalidModel);

    {
        std::ofstream changed_config(artifact_directory.path() / "config.json",
                                     std::ios::binary | std::ios::app);
        changed_config << ' ';
    }
    auto unchanged_manifest_artifact = adapter.parse_model(artifact_package);
    check(static_cast<bool>(unchanged_manifest_artifact));

    {
        std::ofstream changed_index(artifact_directory.path() / "model.safetensors.index.json",
                                    std::ios::binary | std::ios::app);
        changed_index << ' ';
    }
    auto stale_artifact = adapter.parse_model(artifact_package);
    check(!stale_artifact);
    check(stale_artifact.error().code == ErrorCode::InvalidModel);
}

void test_model_neutral_expert_storage()
{
    Qwen3_5MoeModelAdapter adapter;
    auto parsed = adapter.parse_model(qwen3_5_moe_package());
    check(static_cast<bool>(parsed));
    MoeModelDescriptor descriptor = parsed.value();
    descriptor.speculative_layer_count = 0;
    descriptor.speculative_kind = SpeculativeModelKind::None;
    descriptor.speculative_target_layer_ids.clear();
    ScopedTestDirectory directory("ncnn_moe_expert_storage_");
    const auto path = directory.path() / "experts.bin";
    constexpr uint64_t payload_size = 6144;
    const std::vector<uint16_t> payload(payload_size / sizeof(uint16_t), float_to_bfloat16(0.25f));
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload_size));
        check(static_cast<bool>(file));
    }
    auto mapped = MappedFileRange::open(path, 0, payload_size);
    check(static_cast<bool>(mapped));
    const auto owner = mapped.value()->share_data();
    WeightMapping weights;
    uint64_t offset = 0;
    for (uint32_t layer_id = 0; layer_id < descriptor.layers.size(); ++layer_id)
    {
        const MoeDescriptor& moe = descriptor.layers[layer_id].moe;
        for (uint32_t expert = 0; expert < moe.expert_count; ++expert)
        {
            const std::string prefix = "layers." + std::to_string(layer_id) + ".experts." + std::to_string(expert) + ".";
            for (const char* projection : {"gate_up", "down"})
            {
                const bool gate_up = std::string_view(projection) == "gate_up";
                TensorData tensor;
                tensor.dtype = DType::BFloat16;
                tensor.shape = gate_up ? std::vector<uint32_t>{moe.intermediate_size * 2, descriptor.hidden_size}
                                       : std::vector<uint32_t>{descriptor.hidden_size, moe.intermediate_size};
                tensor.mapped_size = tensor.element_count() * sizeof(uint16_t);
                tensor.mapped_data = std::shared_ptr<const uint8_t>(owner, owner.get() + offset);
                offset += tensor.mapped_size;
                weights.emplace(prefix + projection + ".weight", std::move(tensor));
            }
        }
    }
    check(offset == payload_size);
    Option on_demand;
    on_demand.expert_memory_mode = ExpertMemoryMode::OnDemand;
    on_demand.expert_cache_size = 4096;
    constexpr uint64_t physical_memory = UINT64_C(8) * 1024 * 1024 * 1024;
    for (const char* model_type : {"generic_moe", "gpt_oss"})
    {
        MoeModelDescriptor named = descriptor;
        named.model_type = model_type;
        normalize_expert_storage(named, weights);
        for (const LayerDescriptor& layer : named.layers)
            check(has_flag(layer.moe.flags, MoeDescriptorFileBackedExperts));
        const auto plan = plan_model_memory(named, on_demand, physical_memory, false, 0, false, &weights);
        check(static_cast<bool>(plan));
        check(plan.value().selected_mode == ExpertMemoryMode::OnDemand);
        check(plan.value().estimated_expert_size == payload_size);
        check(plan.value().expert_pair_size == 384 && plan.value().expert_cache_size == 4096);
    }
    // The planner also resolves capabilities for callers that pass the raw
    // architectural descriptor, without adapter-specific storage flags.
    check(!has_flag(descriptor.layers.front().moe.flags, MoeDescriptorFileBackedExperts));
    check(static_cast<bool>(plan_model_memory(descriptor, on_demand, physical_memory, false, 0, false, &weights)));

    const std::string gate_up_name = "layers.0.experts.0.gate_up.weight";
    const std::string down_name = "layers.0.experts.0.down.weight";
    const auto reject_incomplete = [&](const WeightMapping& candidate) {
        MoeModelDescriptor stale = descriptor;
        for (LayerDescriptor& layer : stale.layers)
            layer.moe.flags |= MoeDescriptorFileBackedExperts;
        normalize_expert_storage(stale, candidate);
        check(!has_flag(stale.layers.front().moe.flags, MoeDescriptorFileBackedExperts));
        check(has_flag(stale.layers.back().moe.flags, MoeDescriptorFileBackedExperts));
        check(!plan_model_memory(stale, on_demand, physical_memory, false, 0, false, &candidate));
    };
    WeightMapping incomplete = weights;
    incomplete.erase(down_name);
    reject_incomplete(incomplete);
    WeightMapping owned = weights;
    TensorData& owned_gate_up = owned.at(gate_up_name);
    owned_gate_up.bfloat16_data.assign(owned_gate_up.bfloat16_values().begin(), owned_gate_up.bfloat16_values().end());
    owned_gate_up.mapped_data.reset();
    owned_gate_up.mapped_size = 0;
    reject_incomplete(owned);
    WeightMapping mismatched = weights;
    mismatched.at(down_name).shape.front() += 1;
    reject_incomplete(mismatched);
    WeightMapping truncated = weights;
    truncated.at(gate_up_name).mapped_size -= sizeof(uint16_t);
    reject_incomplete(truncated);
    WeightMapping separate = weights;
    TensorData gate = separate.at(gate_up_name);
    gate.shape.front() /= 2;
    gate.mapped_size /= 2;
    separate.erase(gate_up_name);
    separate.emplace("layers.0.experts.0.gate.weight", gate);
    separate.emplace("layers.0.experts.0.up.weight", std::move(gate));
    reject_incomplete(separate);

    MoeModelDescriptor quantized = descriptor;
    quantized.layers.resize(1);
    quantized.hidden_size = 256;
    quantized.intermediate_size = 256;
    quantized.layers.front().moe.intermediate_size = 256;
    quantized.layers.front().moe.expert_weight_dtype = DType::Q4K;
    quantized.layers.front().moe.layout = ExpertLayout::GateUpDown;
    WeightMapping quantized_weights;
    for (uint32_t expert = 0; expert < quantized.layers.front().moe.expert_count; ++expert)
    {
        const std::string prefix = "layers.0.experts." + std::to_string(expert) + ".";
        for (const char* projection : {"gate", "up", "down"})
        {
            TensorData tensor;
            tensor.dtype = std::string_view(projection) == "down" ? DType::Q6K : DType::Q4K;
            tensor.shape = {256, 256};
            auto storage = std::make_shared<MxFp4FileStorage>();
            storage->blocks_path = "expert-storage-qnk.bin";
            storage->blocks_size = qnk_storage_bytes(tensor.dtype, 256, 256);
            tensor.mxfp4_file_storage = std::move(storage);
            quantized_weights.emplace(prefix + projection + ".weight", std::move(tensor));
        }
    }
    normalize_expert_storage(quantized, quantized_weights);
    check(has_flag(quantized.layers.front().moe.flags, MoeDescriptorFileBackedExperts));
    const auto reject_unjoinable = [&](const WeightMapping& candidate) {
        MoeModelDescriptor normalized = quantized;
        normalize_expert_storage(normalized, candidate);
        check(!has_flag(normalized.layers.front().moe.flags, MoeDescriptorFileBackedExperts));
    };
    const std::string up_name = "layers.0.experts.0.up.weight";
    WeightMapping different_columns = quantized_weights;
    different_columns.at(up_name).input_column_indices = {0};
    reject_unjoinable(different_columns);
    WeightMapping secondary_range = quantized_weights;
    auto secondary_storage = std::make_shared<MxFp4FileStorage>(*secondary_range.at(up_name).mxfp4_file_storage);
    secondary_storage->secondary_blocks_path = "secondary-experts.bin";
    secondary_range.at(up_name).mxfp4_file_storage = std::move(secondary_storage);
    reject_unjoinable(secondary_range);
    WeightMapping different_sizes = quantized_weights;
    auto short_storage = std::make_shared<MxFp4FileStorage>(*different_sizes.at(up_name).mxfp4_file_storage);
    --short_storage->blocks_size;
    different_sizes.at(up_name).mxfp4_file_storage = std::move(short_storage);
    reject_unjoinable(different_sizes);

    MoeModelDescriptor with_speculative = descriptor;
    with_speculative.speculative_layer_count = 1;
    normalize_expert_storage(with_speculative, weights);
    check(!has_flag(with_speculative.layers.back().moe.flags, MoeDescriptorFileBackedExperts));
}

static void write_qwen4_mxfp4_test_artifact(ModelPackage& package,
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
        package.manifest.raw_json = std::regex_replace(package.manifest.raw_json,
                                                       std::regex(replacement.first),
                                                       replacement.second);
    }
    const std::string index_json = "{}\n";
    {
        std::ofstream config(root / "config.json", std::ios::binary);
        config << package.manifest.raw_json;
    }
    {
        std::ofstream index(root / "model.safetensors.index.json",
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
        add_tensor(prefix + "gate_up.blocks",
                   {expert_count, intermediate_size * 2, hidden_size / 32, 16});
        add_tensor(prefix + "gate_up.scales",
                   {expert_count, intermediate_size * 2, hidden_size / 32});
        add_tensor(prefix + "down.blocks",
                   {expert_count, hidden_size, intermediate_size / 32, 16});
        add_tensor(prefix + "down.scales",
                   {expert_count, hidden_size, intermediate_size / 32});
    };
    for (uint32_t layer_id = 0; layer_id < layer_count; ++layer_id)
    {
        add_bank("__ncnn_moe_qwen3_8_mxfp4__.layers."
                 + std::to_string(layer_id)
                 + ".experts.");
    }
    header << '}';
    std::string encoded_header = header.str();
    encoded_header.append((8 - encoded_header.size() % 8) % 8,
                          ' ');
    std::ofstream artifact(root / "ncnn-moe-qwen3.8-mxfp4.safetensors",
                           std::ios::binary);
    const uint64_t header_bytes = encoded_header.size();
    artifact.write(reinterpret_cast<const char*>(&header_bytes),
                   sizeof(header_bytes));
    artifact.write(encoded_header.data(),
                   static_cast<std::streamsize>(encoded_header.size()));
    std::array<char, 4096> zeros = {};
    for (uint64_t written = 0; written < data_offset;)
    {
        const uint64_t count = std::min<uint64_t>(zeros.size(),
                                                  data_offset - written);
        artifact.write(zeros.data(),
                       static_cast<std::streamsize>(count));
        written += count;
    }
}

ModelPackage qwen4_exp_package()
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

    ModelPackage nested_text_config = qwen4_exp_package();
    const size_t outer_object_end = nested_text_config.manifest.raw_json.rfind('}');
    check(outer_object_end != std::string::npos);
    nested_text_config.manifest.raw_json.insert(outer_object_end,
                                                R"(,"hidden_size":1,"ple_layer_ids":[0])");
    auto nested_text_config_result = adapter.parse_model(nested_text_config);
    check(static_cast<bool>(nested_text_config_result));
    check(nested_text_config_result.value().hidden_size == 16);
    check(nested_text_config_result.value().layers[1].ple.enabled());

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
    memory_options.expert_memory_mode = ExpertMemoryMode::Eager;
    auto memory = plan_model_memory(descriptor, memory_options, UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(memory));
    check(memory.value().selected_mode == ExpertMemoryMode::Eager);
    check(memory.value().expert_cache_size == 0);
    MoeModelDescriptor without_ple = descriptor;
    without_ple.layers[1].ple = {};
    auto memory_without_ple = plan_model_memory(without_ple, memory_options, UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(memory_without_ple));
    check(memory.value().estimated_dense_size
          == memory_without_ple.value().estimated_dense_size + 3256);
    MoeModelDescriptor large_file_backed_ple = descriptor;
    large_file_backed_ple.layers[1].ple.embedding_row_count = UINT64_C(320001536);
    auto large_ple_memory = plan_model_memory(large_file_backed_ple, memory_options, UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(large_ple_memory));
    check(large_ple_memory.value().estimated_dense_size
          == memory.value().estimated_dense_size);

    const MoeDescriptor& moe = descriptor.layers[0].moe;
    check(moe.expert_count == 4);
    check(moe.top_k == 2);
    check(moe.shared_expert_count == 1);
    check(moe.expert_weight_dtype == DType::BFloat16);
    check(moe.layout == ExpertLayout::PackedGateUpDown);
    check(has_flag(moe.flags, MoeDescriptorSharedExpertGate));
    check(!has_flag(moe.flags, MoeDescriptorFileBackedExperts));
    Option unsupported_on_demand;
    unsupported_on_demand.expert_memory_mode = ExpertMemoryMode::OnDemand;
    check(!plan_model_memory(descriptor, unsupported_on_demand, UINT64_C(8) * 1024 * 1024 * 1024));

    ModelPackage invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(invalid.manifest.raw_json,
                                                   std::regex(R"("output_gate_type"\s*:\s*"sigmoid")"),
                                                   R"("output_gate_type": "silu")");
    check(!adapter.parse_model(invalid));

    invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(invalid.manifest.raw_json,
                                                   std::regex(R"("ple_layer_ids"\s*:\s*\[2\])"),
                                                   R"("ple_layer_ids": [0])");
    check(!adapter.parse_model(invalid));

    invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(invalid.manifest.raw_json,
                                                   std::regex(R"("split_ngram_parts"\s*:\s*2)"),
                                                   R"("split_ngram_parts": 0)");
    check(!adapter.parse_model(invalid));

    invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(invalid.manifest.raw_json,
                                                   std::regex(R"("num_key_value_heads"\s*:\s*1)"),
                                                   R"("num_key_value_heads": 0)");
    check(!adapter.parse_model(invalid));

    for (const char* layer_ids : {"[-1]", "[2.5]", "[2,]"})
    {
        ModelPackage invalid_ple = qwen4_exp_package();
        invalid_ple.manifest.raw_json = std::regex_replace(invalid_ple.manifest.raw_json,
                                                           std::regex(R"("ple_layer_ids"\s*:\s*\[2\])"),
                                                           std::string(R"("ple_layer_ids": )") + layer_ids);
        auto invalid_ple_result = adapter.parse_model(invalid_ple);
        check(!invalid_ple_result);
        check(invalid_ple_result.error().code == ErrorCode::InvalidModel);
    }

    invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(invalid.manifest.raw_json,
                                                   std::regex(R"("rope_parameters"\s*:\s*\{[^{}]*\})"),
                                                   R"("rope_parameters":[1])");
    auto invalid_rope_parameters = adapter.parse_model(invalid);
    check(!invalid_rope_parameters);
    check(invalid_rope_parameters.error().message
          == "invalid Qwen4 Exp object field: rope_parameters");

    for (const char* factor : {"-0.5", "1073741824", "1e38", "0", "0.625", "0.3"})
    {
        ModelPackage invalid_rotary = qwen4_exp_package();
        invalid_rotary.manifest.raw_json = std::regex_replace(invalid_rotary.manifest.raw_json,
                                                              std::regex(R"("partial_rotary_factor"\s*:\s*0.5)"),
                                                              std::string(R"("partial_rotary_factor": )") + factor);
        auto invalid_rotary_result = adapter.parse_model(invalid_rotary);
        check(!invalid_rotary_result);
        check(invalid_rotary_result.error().code == ErrorCode::InvalidModel);
        check(invalid_rotary_result.error().message
              == "unsupported Qwen4 Exp architectural dimensions");
    }

    invalid = qwen4_exp_package();
    invalid.manifest.raw_json = std::regex_replace(invalid.manifest.raw_json,
                                                   std::regex(R"("ple_layer_ids"\s*:\s*\[2\])"),
                                                   R"("ple_layer_ids": [18446744073709551616])");
    auto oversized_ple_layer_ids = adapter.parse_model(invalid);
    check(!oversized_ple_layer_ids);
    check(oversized_ple_layer_ids.error().code == ErrorCode::InvalidModel);
    check(oversized_ple_layer_ids.error().message
          == "invalid Qwen4 Exp integer array: ple_layer_ids");

    ScopedTestDirectory artifact_directory("ncnn_moe_qwen4_artifact_test_");
    ModelPackage artifact_package = qwen4_exp_package();
    write_qwen4_mxfp4_test_artifact(artifact_package,
                                    artifact_directory.path());
    auto artifact_descriptor = adapter.parse_model(artifact_package);
    check(static_cast<bool>(artifact_descriptor));
    check(artifact_descriptor.value().layers[0].moe.expert_weight_dtype
          == DType::MxFp4);
    check(artifact_descriptor.value().layers[3].moe.expert_weight_dtype
          == DType::MxFp4);
    check(!has_flag(artifact_descriptor.value().layers[0].moe.flags,
                    MoeDescriptorFileBackedExperts));
    auto artifact_archive = SafetensorsArchive::open_file(artifact_directory.path() / "ncnn-moe-qwen3.8-mxfp4.safetensors");
    check(static_cast<bool>(artifact_archive));
    WeightMapping artifact_experts;
    for (uint32_t layer_id = 0; layer_id < artifact_descriptor.value().layers.size(); ++layer_id)
    {
        const MoeDescriptor& layer_moe = artifact_descriptor.value().layers[layer_id].moe;
        const std::string source = "__ncnn_moe_qwen3_8_mxfp4__.layers." + std::to_string(layer_id) + ".experts.";
        for (uint32_t expert_id = 0; expert_id < layer_moe.expert_count; ++expert_id)
        {
            const std::string target = "layers." + std::to_string(layer_id) + ".experts." + std::to_string(expert_id) + ".";
            check(static_cast<bool>(add_mxfp4_expert(artifact_experts, artifact_archive.value(), target + "gate_up.weight",
                                                     source + "gate_up.blocks", source + "gate_up.scales", expert_id,
                                                     layer_moe.intermediate_size * 2, artifact_descriptor.value().hidden_size,
                                                     SafetensorLoadDeferMxfp4Data)));
            check(static_cast<bool>(add_mxfp4_expert(artifact_experts, artifact_archive.value(), target + "down.weight",
                                                     source + "down.blocks", source + "down.scales", expert_id,
                                                     artifact_descriptor.value().hidden_size, layer_moe.intermediate_size,
                                                     SafetensorLoadDeferMxfp4Data)));
        }
    }
    MoeModelDescriptor artifact_storage = artifact_descriptor.value();
    normalize_expert_storage(artifact_storage, artifact_experts);
    for (const LayerDescriptor& layer : artifact_storage.layers)
        check(has_flag(layer.moe.flags, MoeDescriptorFileBackedExperts));
    Option artifact_memory;
    artifact_memory.expert_memory_mode = ExpertMemoryMode::OnDemand;
    artifact_memory.expert_cache_size = 4096;
    check(static_cast<bool>(plan_model_memory(artifact_storage, artifact_memory, UINT64_C(8) * 1024 * 1024 * 1024,
                                              false, 0, false, &artifact_experts)));

    ModelPackage changed_manifest = artifact_package;
    changed_manifest.manifest.raw_json += ' ';
    auto changed_manifest_artifact = adapter.parse_model(changed_manifest);
    check(!changed_manifest_artifact);
    check(changed_manifest_artifact.error().code == ErrorCode::InvalidModel);

    {
        std::ofstream changed_config(artifact_directory.path() / "config.json",
                                     std::ios::binary | std::ios::app);
        changed_config << ' ';
    }
    auto unchanged_manifest_artifact = adapter.parse_model(artifact_package);
    check(static_cast<bool>(unchanged_manifest_artifact));

    {
        std::ofstream changed_index(artifact_directory.path() / "model.safetensors.index.json",
                                    std::ios::binary | std::ios::app);
        changed_index << ' ';
    }
    auto stale_artifact = adapter.parse_model(artifact_package);
    check(!stale_artifact);
    check(stale_artifact.error().code == ErrorCode::InvalidModel);
}

static void add_qwen4_mapping_bfloat16(WeightMapping& mapping,
                                       const std::string& name,
                                       std::vector<uint32_t> shape,
                                       float value = 0.0f)
{
    TensorData tensor;
    tensor.dtype = DType::BFloat16;
    tensor.shape = std::move(shape);
    tensor.bfloat16_data.assign(tensor.element_count(), float_to_bfloat16(value));
    check(mapping.emplace(name, std::move(tensor)).second);
}

static void add_qwen4_mapping_int64(WeightMapping& mapping,
                                    const std::string& name,
                                    std::vector<int64_t> values)
{
    TensorData tensor;
    tensor.dtype = DType::Int64;
    tensor.shape = {static_cast<uint32_t>(values.size())};
    tensor.int64_data = std::move(values);
    check(mapping.emplace(name, std::move(tensor)).second);
}

WeightMapping qwen4_exp_test_mapping(const MoeModelDescriptor& descriptor)
{
    WeightMapping mapping;
    add_qwen4_mapping_bfloat16(mapping, "token_embedding.weight",
                               {descriptor.vocabulary_size, descriptor.hidden_size});
    add_qwen4_mapping_bfloat16(mapping, "lm_head.weight",
                               {descriptor.vocabulary_size, descriptor.hidden_size});
    const uint32_t expanded_size = descriptor.hyper_connection_multiplier
                                   * descriptor.hidden_size;
    add_qwen4_mapping_bfloat16(mapping, "gated_residual.head.norm.weight", {expanded_size});
    add_qwen4_mapping_bfloat16(mapping, "gated_residual.head.mix_down.weight",
                               {descriptor.hyper_connection_low_rank, expanded_size});
    add_qwen4_mapping_bfloat16(mapping, "gated_residual.head.mix_up.weight",
                               {expanded_size, descriptor.hyper_connection_low_rank});

    for (uint32_t layer_id = 0;
         layer_id < descriptor.layers.size();
         ++layer_id)
    {
        const std::string layer = "layers." + std::to_string(layer_id) + ".";
        for (const char* block : {"attention", "ffn"})
        {
            const std::string prefix = layer + "gated_residual." + block + ".";
            add_qwen4_mapping_bfloat16(mapping, prefix + "norm.weight", {expanded_size});
            add_qwen4_mapping_bfloat16(mapping, prefix + "mix_down.weight",
                                       {descriptor.hyper_connection_low_rank, expanded_size});
            add_qwen4_mapping_bfloat16(mapping, prefix + "mix_up.weight",
                                       {expanded_size, descriptor.hyper_connection_low_rank});
            add_qwen4_mapping_bfloat16(mapping, prefix + "inject.weight",
                                       {descriptor.hyper_connection_multiplier, expanded_size});
        }
        const MoeDescriptor& moe = descriptor.layers[layer_id].moe;
        add_qwen4_mapping_bfloat16(mapping, layer + "router.weight",
                                   {moe.expert_count, descriptor.hidden_size});
        add_qwen4_mapping_bfloat16(mapping, layer + "shared_expert.gate.weight",
                                   {moe.intermediate_size, descriptor.hidden_size});
        add_qwen4_mapping_bfloat16(mapping, layer + "shared_expert.up.weight",
                                   {moe.intermediate_size, descriptor.hidden_size});
        add_qwen4_mapping_bfloat16(mapping, layer + "shared_expert.down.weight",
                                   {descriptor.hidden_size, moe.intermediate_size});
        add_qwen4_mapping_bfloat16(mapping, layer + "shared_expert.router_gate.weight",
                                   {1, descriptor.hidden_size});
        for (uint32_t expert_id = 0;
             expert_id < moe.expert_count;
             ++expert_id)
        {
            const std::string expert = layer + "experts."
                                       + std::to_string(expert_id) + ".";
            add_qwen4_mapping_bfloat16(mapping, expert + "gate_up.weight",
                                       {moe.intermediate_size * 2, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, expert + "down.weight",
                                       {descriptor.hidden_size, moe.intermediate_size});
        }

        const AttentionDescriptor& attention = descriptor.layers[layer_id].attention;
        if (attention.kind == AttentionKind::GatedDeltaNet)
        {
            const uint32_t key_size = attention.kv_head_count
                                      * attention.head_dimension;
            const uint32_t value_size = attention.head_count
                                        * attention.value_head_dimension;
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.delta.qkv.weight",
                                       {key_size * 2 + value_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.delta.z.weight",
                                       {value_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.delta.beta.weight",
                                       {attention.head_count, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.delta.alpha.weight",
                                       {attention.head_count, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.delta.convolution.weight",
                                       {key_size * 2 + value_size, 1,
                                        attention.convolution_kernel_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.delta.time_bias",
                                       {attention.head_count});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.delta.decay_log",
                                       {attention.head_count});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.delta.norm.weight",
                                       {attention.value_head_dimension});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.output.weight",
                                       {descriptor.hidden_size, value_size});
        }
        else
        {
            const uint32_t query_size = attention.head_count
                                        * attention.head_dimension;
            const uint32_t key_value_size = attention.kv_head_count
                                            * attention.head_dimension;
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.query.weight",
                                       {query_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.key.weight",
                                       {key_value_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.value.weight",
                                       {key_value_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.output.weight",
                                       {descriptor.hidden_size, query_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.output_gate.weight",
                                       {query_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.query_norm.weight",
                                       {attention.head_dimension});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.key_norm.weight",
                                       {attention.head_dimension});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.qsa.query_key.weight",
                                       {(attention.index_head_count + 1)
                                            * attention.index_head_dimension,
                                        descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.qsa.query_norm.weight",
                                       {attention.index_head_dimension});
            add_qwen4_mapping_bfloat16(mapping, layer + "attention.qsa.key_norm.weight",
                                       {attention.index_head_dimension});
        }

        const PleDescriptor& ple = descriptor.layers[layer_id].ple;
        if (ple.enabled())
        {
            add_qwen4_mapping_bfloat16(mapping, layer + "ple.key.weight",
                                       {expanded_size, descriptor.hidden_size});
            add_qwen4_mapping_bfloat16(mapping, layer + "ple.value.weight",
                                       {descriptor.hidden_size, descriptor.hidden_size});
            for (const char* norm : {"key", "query", "convolution"})
            {
                add_qwen4_mapping_bfloat16(mapping, layer + "ple." + norm + "_norm.weight",
                                           {expanded_size});
            }
            add_qwen4_mapping_bfloat16(mapping, layer + "ple.convolution.weight",
                                       {expanded_size, 1, ple.convolution_kernel_size});
            add_qwen4_mapping_int64(mapping, layer + "ple.hash_multipliers", {1, 3, 5});
            add_qwen4_mapping_int64(mapping, layer + "ple.head_vocabulary_sizes", {2, 3});
            add_qwen4_mapping_int64(mapping, layer + "ple.head_offsets", {0, 2});
            const uint32_t head_count = (ple.ngram_size - 1)
                                        * ple.heads_per_ngram;
            const uint32_t head_dimension = ple.embedding_dimension
                                            / head_count;
            for (uint32_t shard = 0;
                 shard < ple.embedding_shard_count;
                 ++shard)
            {
                add_qwen4_mapping_bfloat16(mapping,
                                           layer + "ple.embedding_shard."
                                               + std::to_string(shard),
                                           {4, head_dimension});
            }
        }
    }
    return mapping;
}

static void make_qwen4_routed_experts_file_backed(WeightMapping& mapping)
{
    for (auto& item : mapping)
    {
        if (item.first.find(".experts.") == std::string::npos)
            continue;
        TensorData& tensor = item.second;
        check(tensor.dtype == DType::BFloat16);
        auto storage = std::make_shared<std::vector<uint16_t>>(std::move(tensor.bfloat16_data));
        tensor.bfloat16_data = {};
        tensor.mapped_data = std::shared_ptr<const uint8_t>(storage,
                                                            reinterpret_cast<const uint8_t*>(storage->data()));
        tensor.mapped_size = storage->size() * sizeof(uint16_t);
    }
}

void test_qwen4_exp_compile_and_execute()
{
    BatchWorkspace batch_workspace;
    Qwen4ExpModelAdapter adapter;
    auto parsed = adapter.parse_model(qwen4_exp_package());
    check(static_cast<bool>(parsed));
    WeightMapping mapping = qwen4_exp_test_mapping(parsed.value());
    make_qwen4_routed_experts_file_backed(mapping);
    CompilerOption compiler_opt;
    compiler_opt.flags |= BackendFileBackedExperts;
    CompiledModel compiled;
    auto compile_status = compile_model(parsed.value(),
                                        std::move(mapping),
                                        compiled,
                                        HybridMode::CpuOnly,
                                        compiler_opt);
    if (!compile_status)
    {
        throw std::runtime_error("Qwen4 Exp test compilation failed: "
                                 + compile_status.error().message);
    }
    check(compiled.net.layer_plans.size() == parsed.value().layers.size());
    for (size_t layer_id = 0; layer_id < compiled.net.layer_plans.size(); ++layer_id)
    {
        const CompiledLayerPlan& layer = compiled.net.layer_plans[layer_id];
        check(layer.layer_id == layer_id);
        check(layer.attention.norm_epsilon == compiled.descriptor.norm_epsilon);
        check(layer.attention.kv_cache_dtype == compiled.descriptor.kv_cache_dtype);
    }
    const Blob* delta_cache = nullptr;
    const Blob* full_attention_cache = nullptr;
    for (const Blob& tensor : compiled.net.blobs)
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
    check(compiled.final_norm_weight == invalid_tensor_handle);
    check(compiled.gated_residual_head.norm_weight
          != invalid_tensor_handle);
    check(compiled.net.layer_plans[1].ple.enabled());
    check(has_flag(compiled.net.layer_plans[3].attention.flags,
                   AttentionBlockQsa));
    check(compiled.net.layer_plans[3].attention.kind
          == AttentionKind::Standard);
    check(compiled.net.layer_plans[0].attention.kind
          == AttentionKind::GatedDeltaNet);
    const AttentionBlockPlan& gdn_attention = compiled.net.layer_plans[0].attention;
    check(gdn_attention.fused_delta_input_operator
          == invalid_compiled_operator_handle);
    check(gdn_attention.gated_delta_vulkan_operator
          == invalid_compiled_operator_handle);
    check(!compiled.operators.at_weight(gdn_attention.output_weight).bfloat16);
    check(!compiled.operators.at_weight(gdn_attention.output_weight).linear);
    check(has_flag(compiled.net.layer_plans[0].attention.flags,
                   AttentionBlockSigmoidGate));
    check(has_flag(compiled.net.layer_plans[0].attention.flags,
                   AttentionBlockExternalResidual));
    const uint64_t expert_pair_size = UINT64_C(3) * parsed.value().intermediate_size
                                      * parsed.value().hidden_size * sizeof(uint16_t);
    compiled.expert_cache = std::make_shared<ExpertCache>(expert_pair_size);

    SessionState state;
    SessionStatistics statistics;
    const std::array<int32_t, 4> prompt = {1, 2, 9, 3};
    auto prefilled_status = forward_model(compiled, prompt, statistics, state, 0);
    if (!prefilled_status)
    {
        throw std::runtime_error("Qwen4 Exp test execution failed: "
                                 + prefilled_status.error().message);
    }
    check(state.logits.rows() == prompt.size());
    for (size_t row = 0; row < state.logits.rows(); ++row)
    {
        check(state.logits.columns() == parsed.value().vocabulary_size);
        for (uint32_t column = 0; column < state.logits.columns(); ++column)
            check_near(state.logits.row(row)[column], 0.0f, 1e-6f);
    }
    check(state.layers[1].ple_token_history
          == std::vector<int32_t>({9, 3}));
    const AttentionBlockPlan& qsa_layer_plan = compiled.net.layer_plans[3].attention;
    check(state.layers[3].qsa_block_keys.size()
          == (prompt.size() / qsa_layer_plan.compression_ratio)
                 * qsa_layer_plan.index_head_dimension);
    check(state.layers[3].qsa_index_key_tail.size()
          == (prompt.size() % qsa_layer_plan.compression_ratio)
                 * qsa_layer_plan.index_head_dimension);
    const ExpertCacheStatistics prefill_cache = compiled.expert_cache->statistics();
    check(prefill_cache.misses > 0);
    check(prefill_cache.bytes_read > 0);
    check(prefill_cache.resident_size <= expert_pair_size);

    const uint64_t prefill_cache_misses = prefill_cache.misses;
    std::vector<float> decoded_logits;
    const std::array<DecodeBatchEntry, 1> entries = {{
        4,
        &statistics,
        &state,
        prompt.size(),
        &decoded_logits,
    }};
    auto decoded = forward_decode_batch(compiled, entries, batch_workspace);
    check(static_cast<bool>(decoded));
    check(decoded_logits.size() == compiled.descriptor.vocabulary_size);
    check(state.layers[1].ple_token_history
          == std::vector<int32_t>({3, 4}));
    check(state.layers[3].qsa_block_keys.size()
          == ((prompt.size() + 1) / qsa_layer_plan.compression_ratio)
                 * qsa_layer_plan.index_head_dimension);
    check(state.layers[3].qsa_index_key_tail.size()
          == ((prompt.size() + 1) % qsa_layer_plan.compression_ratio)
                 * qsa_layer_plan.index_head_dimension);
    check(compiled.expert_cache->statistics().misses > prefill_cache_misses);

    MoeModelDescriptor sliding_qsa = parsed.value();
    sliding_qsa.layers[3].attention.sliding_window = 4;
    CompiledModel rejected;
    auto rejected_status = compile_model(sliding_qsa,
                                         qwen4_exp_test_mapping(sliding_qsa),
                                         rejected,
                                         HybridMode::CpuOnly);
    check(!rejected_status);
    check(rejected_status.error().code == ErrorCode::UnsupportedModel);

    WeightMapping missing_query_norm_mapping = qwen4_exp_test_mapping(parsed.value());
    missing_query_norm_mapping.erase("layers.3.attention.query_norm.weight");
    missing_query_norm_mapping.erase("layers.3.attention.key.weight");
    CompiledModel missing_query_norm;
    auto missing_query_norm_status = compile_model(parsed.value(),
                                                   std::move(missing_query_norm_mapping),
                                                   missing_query_norm,
                                                   HybridMode::CpuOnly);
    check(!missing_query_norm_status);
    check(missing_query_norm_status.error().code == ErrorCode::InvalidModel);
    check(missing_query_norm_status.error().message
          == "missing tensor: layers.3.attention.query_norm.weight");

    WeightMapping missing_qsa_query_mapping = qwen4_exp_test_mapping(sliding_qsa);
    missing_qsa_query_mapping.erase("layers.3.attention.query.weight");
    CompiledModel missing_qsa_query;
    auto missing_qsa_query_status = compile_model(sliding_qsa,
                                                  std::move(missing_qsa_query_mapping),
                                                  missing_qsa_query,
                                                  HybridMode::CpuOnly);
    check(!missing_qsa_query_status);
    check(missing_qsa_query_status.error().code == ErrorCode::UnsupportedModel);
    check(missing_qsa_query_status.error().message
          == "QSA attention does not support a sliding-window KV cache");
}

void test_qwen4_exp_vulkan_projections()
{
    if (get_gpu_count() == 0)
        return;

    Qwen4ExpModelAdapter adapter;
    auto parsed = adapter.parse_model(qwen4_exp_package());
    check(static_cast<bool>(parsed));
    for (bool qsa_attention : {true, false})
    {
        MoeModelDescriptor descriptor = parsed.value();
        AttentionDescriptor& attention = descriptor.layers[3].attention;
        if (qsa_attention)
        {
            attention.index_top_k = 1;
            attention.index_token_budget = attention.compression_ratio;
        }
        else
        {
            attention.flags &= ~(AttentionDescriptorQsa | AttentionDescriptorQueryKeyNorm | AttentionDescriptorOutputGate);
            attention.qk_rope_head_dimension = attention.head_dimension;
            descriptor.norm_weight_offset = 0.0f;
        }
        WeightMapping mapping = qwen4_exp_test_mapping(descriptor);
        for (const char* name : {"query", "key", "value", "output", "output_gate", "qsa.query_key"})
        {
            TensorData& tensor = mapping.at(std::string("layers.3.attention.") + name + ".weight");
            for (size_t index = 0; index < tensor.bfloat16_data.size(); ++index)
            {
                const int value = static_cast<int>((index * 7 + 3) % 17) - 8;
                tensor.bfloat16_data[index] = float_to_bfloat16(static_cast<float>(value) / 32.0f);
            }
        }
        CompiledModel cpu_model;
        check(static_cast<bool>(compile_model(descriptor, mapping, cpu_model, HybridMode::CpuOnly)));
        const AttentionBlockPlan& cpu_plan = cpu_model.net.layer_plans[3].attention;
        ActivationBuffer input(13, descriptor.hidden_size);
        for (size_t row = 0; row < input.rows(); ++row)
            for (uint32_t column = 0; column < input.columns(); ++column)
                input.row(row)[column] = static_cast<float>(static_cast<int>((row * 5 + column * 3) % 19) - 9) / 8.0f;

        LayerCache cpu_cache;
        AttentionScratch cpu_scratch;
        ActivationBuffer cpu_output;
        check(static_cast<bool>(forward_attention(cpu_model.weights, cpu_model.operators, cpu_plan,
                                                  ExecutionBackend::Cpu, 0, cpu_cache, cpu_scratch,
                                                  input, cpu_output, cpu_model.opt.optimization_flags)));
        check(cpu_cache.token_count == input.rows());
        check(cpu_output.rows() == input.rows());

        // These Qwen blocks retain external residual semantics. Without the
        // complete device chain they are planned on CPU before execution.
        CompilerOption opt;
        opt.flags |= BackendVulkanDense | BackendVulkanAttention;
        opt.device_index = get_default_gpu_index();
        opt.vulkan_runtime = create_vulkan_runtime();
        CompiledModel hybrid_model;
        auto compiled = compile_model(descriptor, mapping, hybrid_model, HybridMode::HybridExperts, opt);
        if (!compiled)
            throw std::runtime_error("projection-only Qwen preparation failed: " + compiled.error().message);
        const AttentionBlockPlan& selected_plan = hybrid_model.net.layer_plans[3].attention;
        check(has_flag(selected_plan.flags, AttentionBlockExternalResidual));
        check(!support_vulkan_attention(hybrid_model.operators, selected_plan));
        const auto attention_node = std::find_if(hybrid_model.net.layers.begin(), hybrid_model.net.layers.end(), [](const Layer& node) {
            return node.kind == LayerKind::Attention && node.plan_index == 3;
        });
        if (attention_node == hybrid_model.net.layers.end() || attention_node->backend != ExecutionBackend::Cpu)
            throw std::runtime_error("projection-only Qwen Attention must have explicit CPU placement (QSA="
                                     + std::to_string(qsa_attention) + ", flags=" + std::to_string(selected_plan.flags) + ")");
        const VulkanStatistics before = get_vulkan_statistics(hybrid_model.vulkan_runtime);
        LayerCache selected_cache;
        AttentionScratch selected_scratch;
        ActivationBuffer selected_output;
        auto selected_prefill = forward_attention(hybrid_model.weights, hybrid_model.operators, selected_plan,
                                                  attention_node->backend, 0, selected_cache, selected_scratch,
                                                  input, selected_output, hybrid_model.opt.optimization_flags);
        if (!selected_prefill)
            throw std::runtime_error("planned Qwen CPU Attention prefill failed: " + selected_prefill.error().message);
        check(selected_cache.token_count == cpu_cache.token_count);
        check(selected_output.rows() == cpu_output.rows() && selected_output.columns() == cpu_output.columns());
        for (size_t index = 0; index < cpu_output.values().size(); ++index)
            check_near(selected_output.values()[index], cpu_output.values()[index], 1e-5f);

        ActivationBuffer continuation(1, descriptor.hidden_size);
        for (uint32_t column = 0; column < continuation.columns(); ++column)
            continuation.row(0)[column] = input.row(0)[column] * 0.5f;
        check(static_cast<bool>(forward_attention(cpu_model.weights, cpu_model.operators, cpu_plan,
                                                  ExecutionBackend::Cpu, input.rows(), cpu_cache, cpu_scratch,
                                                  continuation, cpu_output, cpu_model.opt.optimization_flags)));
        auto selected_decode = forward_attention(hybrid_model.weights, hybrid_model.operators, selected_plan,
                                                 attention_node->backend, input.rows(), selected_cache, selected_scratch,
                                                 continuation, selected_output, hybrid_model.opt.optimization_flags);
        if (!selected_decode)
            throw std::runtime_error("planned Qwen CPU Attention decode failed: " + selected_decode.error().message);
        check(selected_cache.token_count == input.rows() + 1 && selected_cache.token_count == cpu_cache.token_count);
        check(selected_output.rows() == cpu_output.rows() && selected_output.columns() == cpu_output.columns());
        for (size_t index = 0; index < cpu_output.values().size(); ++index)
            check_near(selected_output.values()[index], cpu_output.values()[index], 1e-5f);
        const VulkanStatistics after = get_vulkan_statistics(hybrid_model.vulkan_runtime);
        check(after.attention_blocks == before.attention_blocks);
        check(after.attention_cpu_fallbacks == before.attention_cpu_fallbacks);

        LayerCache unavailable_cache;
        AttentionScratch unavailable_scratch;
        ActivationBuffer unavailable_output;
        const auto unavailable = forward_attention(hybrid_model.weights, hybrid_model.operators, selected_plan,
                                                   ExecutionBackend::Vulkan, 0, unavailable_cache, unavailable_scratch,
                                                   input, unavailable_output, hybrid_model.opt.optimization_flags);
        check(!unavailable && unavailable.error().code == ErrorCode::InternalError);
        check(unavailable.error().message.find("without a complete attention operator") != std::string::npos);
        check(unavailable_cache.token_count == 0 && unavailable_output.rows() == 0);
    }
}

} // namespace moe
} // namespace ncnn
