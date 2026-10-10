#include "kernels/attention.h"
#include "kernels/activationbuffer.h"
#include "kernels/mxfp4.h"
#include "kernels/ops.h"
#include "kernels/statecache.h"
#include "engine/cpu.h"
#include "graph/compiledoperator.h"
#include "graph/graph.h"
#include "ncnn/moe/option.h"
#include "storage/weightstore.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

static void scratch_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(std::string("scratch test failed: ") + message);
}

static void scratch_check_near(float actual, float expected, float tolerance, const char* message)
{
    if (!(std::abs(actual - expected) <= tolerance))
        throw std::runtime_error(std::string("scratch test mismatch: ") + message);
}

#if defined(_OPENMP)
struct ScratchOpenMpDynamicScope
{
    const int previous = omp_get_dynamic();

    ScratchOpenMpDynamicScope()
    {
        omp_set_dynamic(0);
    }

    ~ScratchOpenMpDynamicScope()
    {
        omp_set_dynamic(previous);
    }
};
#endif

static TensorData make_mxfp4_tensor(uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData tensor;
    tensor.dtype = DType::MxFp4;
    tensor.shape = {rows, columns};
    tensor.mxfp4_scales.resize(static_cast<size_t>(rows) * columns / 32);
    tensor.mxfp4_blocks.resize(static_cast<size_t>(rows) * columns / 2);
    for (size_t index = 0; index < tensor.mxfp4_scales.size(); ++index)
        tensor.mxfp4_scales[index] = static_cast<uint8_t>(122 + (seed + index) % 7);
    for (size_t index = 0; index < tensor.mxfp4_blocks.size(); ++index)
    {
        const uint8_t low = static_cast<uint8_t>((seed + index * 3) % 16);
        const uint8_t high = static_cast<uint8_t>((seed * 5 + index * 7 + 1) % 16);
        tensor.mxfp4_blocks[index] = static_cast<uint8_t>(low | (high << 4));
    }
    return tensor;
}

static void check_mxfp4_outputs(const ActivationBuffer& actual,
                                const ActivationBuffer& expected,
                                float tolerance)
{
    scratch_check(actual.rows() == expected.rows(), "MXFP4 output row count");
    scratch_check(actual.columns() == expected.columns(), "MXFP4 output column count");
    for (size_t row = 0; row < actual.rows(); ++row)
    {
        for (uint32_t column = 0; column < actual.columns(); ++column)
            scratch_check_near(actual.row(row)[column], expected.row(row)[column], tolerance, "MXFP4 output");
    }
}

static void test_mxfp4_scratch_reuse()
{
    const TensorData gate_up = make_mxfp4_tensor(64, 32, 3);
    const TensorData down = make_mxfp4_tensor(7, 32, 11);
    ActivationBuffer input_zero(1, 32);
    ActivationBuffer input_one(1, 32);
    ActivationBuffer input_two(1, 32);
    for (uint32_t column = 0; column < input_zero.columns(); ++column)
    {
        input_zero.row(0)[column] = static_cast<float>(static_cast<int>(column % 13) - 6) * 0.03125f;
        input_one.row(0)[column] = static_cast<float>(static_cast<int>((column * 3) % 17) - 8) * 0.015625f;
        input_two.row(0)[column] = static_cast<float>(static_cast<int>((column * 5) % 19) - 9) * 0.015625f;
    }

    std::array<Mxfp4Task, 3> tasks;
    std::array<ActivationBuffer, 3> outputs;
    const std::array<const ActivationBuffer*, 3> inputs = {&input_zero, &input_one, &input_two};
    for (size_t index = 0; index < tasks.size(); ++index)
    {
        tasks[index].gate_up = &gate_up;
        tasks[index].down = &down;
        tasks[index].input = inputs[index];
        tasks[index].output = &outputs[index];
        tasks[index].activation = ExpertActivation::DeepSeekSwiGlu;
        tasks[index].activation_limit = 5.25f;
    }

    const uint64_t q8_flags = OptimizationDefaultFlags | OptimizationCpuPackedWeights;
    const uint64_t scalar_flags = q8_flags & ~OptimizationCpuMxfp4Q8;

    std::array<ActivationBuffer, 3> disabled_outputs;
    std::array<Mxfp4Task, 3> disabled_tasks = tasks;
    for (size_t index = 0; index < disabled_tasks.size(); ++index)
        disabled_tasks[index].output = &disabled_outputs[index];
    Mxfp4Scratch disabled_scratch;
    scratch_check(forward_experts_mxfp4(std::span<const Mxfp4Task>(disabled_tasks.data(), disabled_tasks.size()),
                                        &disabled_scratch,
                                        scalar_flags),
                  "MXFP4 Q8-disabled batch");
    scratch_check(std::all_of(disabled_scratch.q8_down_enabled.begin(), disabled_scratch.q8_down_enabled.end(), [](uint8_t value) { return value == 0; }), "Q8-disabled down bookkeeping");
    scratch_check(std::all_of(disabled_scratch.q8_gate_packed.begin(), disabled_scratch.q8_gate_packed.end(), [](uint8_t value) { return value == 0; }), "Q8-disabled gate bookkeeping");
    for (const auto& packed : disabled_scratch.q8_down_packed)
        scratch_check(!packed, "Q8-disabled packed bookkeeping");

    Mxfp4Scratch scratch;
    scratch_check(forward_experts_mxfp4(std::span<const Mxfp4Task>(tasks.data(), tasks.size()),
                                        &scratch,
                                        q8_flags),
                  "MXFP4 initial batch");
    std::array<const std::byte*, 3> activated_storage = {};
    for (size_t index = 0; index < tasks.size(); ++index)
        activated_storage[index] = scratch.activated[index].bytes().data();
    const size_t activated_size = scratch.activated.size();
    const size_t q8_enabled_size = scratch.q8_down_enabled.size();
    const size_t q8_packed_size = scratch.q8_down_packed.size();
    for (const auto& packed : scratch.q8_down_packed)
        scratch_check(!packed, "packed sidecar reference cleanup");

    std::array<ActivationBuffer, 3> fresh_outputs;
    std::array<Mxfp4Task, 3> fresh_tasks = tasks;
    for (size_t index = 0; index < fresh_tasks.size(); ++index)
        fresh_tasks[index].output = &fresh_outputs[index];
    Mxfp4Scratch fresh_scratch;
    scratch_check(forward_experts_mxfp4(std::span<const Mxfp4Task>(fresh_tasks.data(), fresh_tasks.size()),
                                        &fresh_scratch,
                                        q8_flags),
                  "MXFP4 fresh Q8 batch");
    for (size_t index = 0; index < tasks.size(); ++index)
        check_mxfp4_outputs(outputs[index], fresh_outputs[index], 1e-5f);

    std::array<ActivationBuffer, 3> expected;
    for (size_t index = 0; index < tasks.size(); ++index)
    {
        Mxfp4Task reference = tasks[index];
        reference.output = &expected[index];
        scratch_check(forward_experts_mxfp4(std::span<const Mxfp4Task>(&reference, 1),
                                            nullptr,
                                            scalar_flags),
                      "MXFP4 reference batch");
        check_mxfp4_outputs(outputs[index], expected[index], 2.0f);
    }

    const uint8_t* q8_enabled_data = scratch.q8_down_enabled.data();
    const uint8_t* q8_gate_data = scratch.q8_gate_packed.data();
    const auto* q8_packed_data = scratch.q8_down_packed.data();
    const size_t q8_gate_size = scratch.q8_gate_packed.size();
    const size_t q8_enabled_capacity = scratch.q8_down_enabled.capacity();
    const size_t q8_gate_capacity = scratch.q8_gate_packed.capacity();
    const size_t q8_packed_capacity = scratch.q8_down_packed.capacity();
    Mxfp4Task disabled_reuse_task = tasks[0];
    scratch_check(forward_experts_mxfp4(std::span<const Mxfp4Task>(&disabled_reuse_task, 1),
                                        &scratch,
                                        scalar_flags),
                  "MXFP4 reused Q8-disabled batch");
    scratch_check(scratch.q8_down_enabled.data() == q8_enabled_data,
                  "Q8-disabled bookkeeping pointer reuse");
    scratch_check(scratch.q8_gate_packed.data() == q8_gate_data,
                  "Q8-disabled gate pointer reuse");
    scratch_check(scratch.q8_down_packed.data() == q8_packed_data,
                  "Q8-disabled packed pointer reuse");
    scratch_check(scratch.q8_gate_packed.size() == q8_gate_size,
                  "Q8-disabled gate size reuse");
    scratch_check(scratch.q8_down_enabled.capacity() == q8_enabled_capacity,
                  "Q8-disabled bookkeeping capacity reuse");
    scratch_check(scratch.q8_gate_packed.capacity() == q8_gate_capacity,
                  "Q8-disabled gate capacity reuse");
    scratch_check(scratch.q8_down_packed.capacity() == q8_packed_capacity,
                  "Q8-disabled packed capacity reuse");
    for (const auto& packed : scratch.q8_down_packed)
        scratch_check(!packed, "packed sidecar reference cleanup after disabled call");

    ActivationBuffer fresh_shrink_output;
    Mxfp4Task fresh_shrink_task = tasks[0];
    fresh_shrink_task.output = &fresh_shrink_output;
    Mxfp4Scratch fresh_shrink_scratch;
    scratch_check(forward_experts_mxfp4(std::span<const Mxfp4Task>(&fresh_shrink_task, 1),
                                        &fresh_shrink_scratch,
                                        q8_flags),
                  "MXFP4 fresh Q8 shrink batch");
    scratch_check(forward_experts_mxfp4(std::span<const Mxfp4Task>(tasks.data(), 1),
                                        &scratch,
                                        q8_flags),
                  "MXFP4 shrink batch");
    check_mxfp4_outputs(outputs[0], fresh_shrink_output, 1e-5f);
    scratch_check(scratch.activated.size() == activated_size, "MXFP4 scratch size after shrink");
    scratch_check(scratch.q8_down_enabled.size() == q8_enabled_size, "Q8 bookkeeping after shrink");
    scratch_check(scratch.q8_down_packed.size() == q8_packed_size, "Q8 packed storage after shrink");
    scratch_check(scratch.activated[0].bytes().data() == activated_storage[0], "MXFP4 storage after shrink");

    std::array<ActivationBuffer, 3> regrow_fresh_outputs;
    std::array<Mxfp4Task, 3> regrow_fresh_tasks = tasks;
    for (size_t index = 0; index < regrow_fresh_tasks.size(); ++index)
        regrow_fresh_tasks[index].output = &regrow_fresh_outputs[index];
    Mxfp4Scratch regrow_fresh_scratch;
    scratch_check(forward_experts_mxfp4(std::span<const Mxfp4Task>(regrow_fresh_tasks.data(), regrow_fresh_tasks.size()),
                                        &regrow_fresh_scratch,
                                        q8_flags),
                  "MXFP4 fresh Q8 regrow batch");
    for (size_t index = 0; index < tasks.size(); ++index)
        tasks[index].output = &outputs[index];
    scratch_check(forward_experts_mxfp4(std::span<const Mxfp4Task>(tasks.data(), tasks.size()),
                                        &scratch,
                                        q8_flags),
                  "MXFP4 regrow batch");
    scratch_check(scratch.activated.size() == activated_size, "MXFP4 scratch size after regrow");
    for (size_t index = 0; index < tasks.size(); ++index)
    {
        scratch_check(scratch.activated[index].bytes().data() == activated_storage[index],
                      "MXFP4 storage after regrow");
        check_mxfp4_outputs(outputs[index], regrow_fresh_outputs[index], 1e-5f);
    }
    for (const auto& packed : scratch.q8_down_packed)
        scratch_check(!packed, "packed sidecar reference cleanup after regrow");

    if (mxfp4_q8_packed_kernel_available())
    {
        CompiledOperator owner;
        tasks[0].down_operator = &owner;
        scratch_check(forward_experts_mxfp4(std::span<const Mxfp4Task>(tasks.data(), 1),
                                            &scratch,
                                            q8_flags),
                      "MXFP4 operator-owned packed batch");
        if (scratch.q8_down_enabled[0])
        {
            scratch_check(owner.mxfp4_q8_packed != nullptr, "operator packed sidecar creation");
            scratch_check(owner.mxfp4_q8_packed.use_count() == 1,
                          "scratch does not retain operator packed sidecar");
        }
        else
        {
            scratch_check(!owner.mxfp4_q8_packed, "FP32 row-pair path avoids packed sidecar");
        }
        check_mxfp4_outputs(outputs[0], regrow_fresh_outputs[0], 1e-5f);
    }
}

static TensorData make_float_matrix(uint32_t rows, uint32_t columns, float diagonal_scale)
{
    TensorData tensor;
    tensor.dtype = DType::Float32;
    tensor.shape = {rows, columns};
    tensor.float32_data.resize(static_cast<size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            tensor.float32_data[static_cast<size_t>(row) * columns + column] = row == column ? diagonal_scale : static_cast<float>(static_cast<int>((row + 3 * column) % 11) - 5) * 0.03125f;
        }
    }
    return tensor;
}

static TensorHandle add_attention_weight(WeightStore& weights,
                                         const char* name,
                                         TensorData tensor)
{
    auto handle = weights.add(name, std::move(tensor));
    scratch_check(static_cast<bool>(handle), "attention weight insertion");
    return handle.value();
}

static ActivationBuffer make_attention_hidden(size_t rows, uint32_t columns = 4)
{
    ActivationBuffer hidden(rows, columns);
    for (size_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            const uint32_t sample = static_cast<uint32_t>((row * 17 + column * 11) % 31);
            hidden.row(row)[column] = static_cast<float>(static_cast<int>(sample) - 15) * 0.015625f;
        }
    }
    return hidden;
}

static AttentionBlockPlan make_attention_plan(WeightStore& weights)
{
    AttentionBlockPlan plan;
    plan.kind = AttentionKind::Standard;
    plan.head_count = 2;
    plan.kv_head_count = 1;
    plan.head_dimension = 2;
    plan.value_head_dimension = 2;
    plan.query_weight = add_attention_weight(weights, "query", make_float_matrix(4, 4, 1.0f));
    plan.key_weight = add_attention_weight(weights, "key", make_float_matrix(2, 4, 0.75f));
    plan.value_weight = add_attention_weight(weights, "value", make_float_matrix(2, 4, 0.5f));
    plan.output_weight = add_attention_weight(weights, "output", make_float_matrix(4, 4, 1.0f));
    return plan;
}

static AttentionBlockPlan make_split_attention_plan(WeightStore& weights)
{
    constexpr uint32_t hidden_size = 256;
    constexpr uint32_t head_count = 4;
    constexpr uint32_t kv_head_count = 2;
    constexpr uint32_t head_dimension = 64;
    AttentionBlockPlan plan;
    plan.kind = AttentionKind::Standard;
    plan.head_count = head_count;
    plan.kv_head_count = kv_head_count;
    plan.head_dimension = head_dimension;
    plan.value_head_dimension = head_dimension;
    plan.rope_head_dimension = head_dimension;
    plan.query_weight = add_attention_weight(weights,
                                             "split_query",
                                             make_float_matrix(hidden_size, hidden_size, 1.0f));
    plan.key_weight = add_attention_weight(weights,
                                           "split_key",
                                           make_float_matrix(kv_head_count * head_dimension,
                                                             hidden_size,
                                                             0.75f));
    plan.value_weight = add_attention_weight(weights,
                                             "split_value",
                                             make_float_matrix(kv_head_count * head_dimension,
                                                               hidden_size,
                                                               0.5f));
    plan.output_weight = add_attention_weight(weights,
                                              "split_output",
                                              make_float_matrix(hidden_size, hidden_size, 1.0f));
    return plan;
}

static LayerCache make_float_attention_cache(uint32_t columns,
                                             uint64_t token_count,
                                             uint64_t capacity_tokens,
                                             uint64_t first_slot)
{
    LayerCache cache;
    cache.columns = columns;
    cache.dtype = DType::Float32;
    cache.token_count = token_count;
    cache.capacity_tokens = capacity_tokens;
    cache.first_slot = first_slot;
    cache.keys.resize(static_cast<size_t>(capacity_tokens) * columns);
    cache.values.resize(static_cast<size_t>(capacity_tokens) * columns);
    for (uint64_t token = 0; token < token_count; ++token)
    {
        const uint64_t slot = (first_slot + token) % capacity_tokens;
        for (uint32_t column = 0; column < columns; ++column)
        {
            const uint32_t key_sample = static_cast<uint32_t>((token * 13 + column * 7) % 37);
            const uint32_t value_sample = static_cast<uint32_t>((token * 5 + column * 11) % 29);
            const size_t offset = static_cast<size_t>(slot) * columns + column;
            cache.keys[offset] = static_cast<float>(static_cast<int>(key_sample) - 18) * 0.015625f;
            cache.values[offset] = static_cast<float>(static_cast<int>(value_sample) - 14) * 0.03125f;
        }
    }
    return cache;
}

static void run_attention(const WeightStore& weights,
                          const CompiledOperatorTable& operators,
                          const AttentionBlockPlan& plan,
                          uint64_t position,
                          LayerCache& cache,
                          AttentionScratch& scratch,
                          const ActivationBuffer& hidden,
                          ActivationBuffer& output,
                          uint64_t flags)
{
    auto result = forward_attention(weights,
                                    operators,
                                    plan,
                                    ExecutionBackend::Cpu,
                                    position,
                                    cache,
                                    scratch,
                                    hidden,
                                    output,
                                    flags);
    scratch_check(static_cast<bool>(result), "attention execution");
}

static void test_attention_scratch_reuse()
{
#if defined(_OPENMP)
    ScratchOpenMpDynamicScope dynamic_scope;
#endif
    CpuOpenMpThreadLimitScope thread_limit_scope;
    thread_limit_scope.set(std::min(4u, std::max(1u, get_physical_cpu_count())));
    const uint32_t attention_test_thread_count = cpu_linear_num_threads();

    WeightStore weights;
    const AttentionBlockPlan plan = make_attention_plan(weights);
    const AttentionBlockPlan split_plan = make_split_attention_plan(weights);
    CompiledOperatorTable operators;
    operators.bind_weight_count(weights.size());
    const uint64_t reference_flags = OptimizationDefaultFlags
                                     & ~OptimizationCpuFlashAttention
                                     & ~OptimizationCpuSplitKvAttention;
    const uint64_t accelerated_flags = OptimizationDefaultFlags;

    AttentionScratch scratch;
    LayerCache short_cache;
    const ActivationBuffer short_hidden = make_attention_hidden(2);
    ActivationBuffer short_output;
    run_attention(weights,
                  operators,
                  plan,
                  0,
                  short_cache,
                  scratch,
                  short_hidden,
                  short_output,
                  reference_flags);
    scratch_check(scratch.logits.size() == short_cache.token_count,
                  "fallback logits size");

    const ActivationBuffer long_hidden = make_attention_hidden(80);
    LayerCache reference_cache;
    AttentionScratch reference_scratch;
    ActivationBuffer reference_long_output;
    run_attention(weights,
                  operators,
                  plan,
                  0,
                  reference_cache,
                  reference_scratch,
                  long_hidden,
                  reference_long_output,
                  reference_flags);

    LayerCache accelerated_cache;
    ActivationBuffer accelerated_long_output;
    AttentionScratch flash_scratch;
    run_attention(weights,
                  operators,
                  plan,
                  0,
                  accelerated_cache,
                  flash_scratch,
                  long_hidden,
                  accelerated_long_output,
                  accelerated_flags);
    check_mxfp4_outputs(accelerated_long_output, reference_long_output, 1e-4f);
    scratch_check(flash_scratch.logits.empty(), "cold flash leaves logits untouched");
    scratch_check(flash_scratch.logits.capacity() == 0, "cold flash does not allocate logits");
    scratch_check(!flash_scratch.workspace.empty(), "flash worker storage allocation");
    const float* worker_data = flash_scratch.workspace.data();
    const size_t worker_capacity = flash_scratch.workspace.capacity();
    const size_t flash_attention_rows = flash_scratch.attention.rows();
    const uint32_t flash_attention_columns = flash_scratch.attention.columns();
    const std::byte* flash_attention_data = flash_scratch.attention.bytes().data();
    const std::vector<float> flash_attention_reference(flash_scratch.attention.values().begin(),
                                                       flash_scratch.attention.values().end());
    for (size_t row = 0; row < flash_scratch.attention.rows(); ++row)
    {
        std::fill_n(flash_scratch.attention.row(row),
                    flash_scratch.attention.columns(),
                    std::numeric_limits<float>::quiet_NaN());
    }
    LayerCache poisoned_flash_cache;
    ActivationBuffer poisoned_flash_output;
    run_attention(weights,
                  operators,
                  plan,
                  0,
                  poisoned_flash_cache,
                  flash_scratch,
                  long_hidden,
                  poisoned_flash_output,
                  accelerated_flags);
    scratch_check(flash_scratch.attention.rows() == flash_attention_rows,
                  "flash output scratch row reuse");
    scratch_check(flash_scratch.attention.columns() == flash_attention_columns,
                  "flash output scratch column reuse");
    scratch_check(flash_scratch.attention.bytes().data() == flash_attention_data,
                  "flash output scratch storage reuse");
    scratch_check(poisoned_flash_output.rows() == accelerated_long_output.rows(),
                  "flash output row count after poison");
    scratch_check(poisoned_flash_output.columns() == accelerated_long_output.columns(),
                  "flash output column count after poison");
    for (size_t row = 0; row < flash_attention_rows; ++row)
    {
        for (uint32_t column = 0; column < flash_attention_columns; ++column)
        {
            scratch_check(flash_scratch.attention.row(row)[column]
                              == flash_attention_reference[row * flash_attention_columns + column],
                          "flash attention output after poisoned scratch");
        }
    }
    for (size_t row = 0; row < accelerated_long_output.rows(); ++row)
    {
        for (uint32_t column = 0; column < accelerated_long_output.columns(); ++column)
        {
            scratch_check(poisoned_flash_output.row(row)[column]
                              == accelerated_long_output.row(row)[column],
                          "flash final output after poisoned scratch");
        }
    }

    LayerCache future_cache = make_float_attention_cache(plan.kv_head_count * plan.head_dimension,
                                                         80, 160, 0);
    future_cache.start_position = 1000;
    for (size_t row = 0; row < flash_scratch.attention.rows(); ++row)
    {
        std::fill_n(flash_scratch.attention.row(row),
                    flash_scratch.attention.columns(),
                    std::numeric_limits<float>::quiet_NaN());
    }
    ActivationBuffer future_output;
    run_attention(weights,
                  operators,
                  plan,
                  0,
                  future_cache,
                  flash_scratch,
                  long_hidden,
                  future_output,
                  accelerated_flags);
    for (size_t row = 0; row < flash_scratch.attention.rows(); ++row)
    {
        for (uint32_t column = 0; column < flash_scratch.attention.columns(); ++column)
        {
            const float value = flash_scratch.attention.row(row)[column];
            scratch_check(std::isfinite(value) && value == 0.0f && !std::signbit(value),
                          "all-future flash attention output is positive zero");
        }
    }
    for (size_t row = 0; row < future_output.rows(); ++row)
    {
        for (uint32_t column = 0; column < future_output.columns(); ++column)
        {
            const float value = future_output.row(row)[column];
            scratch_check(std::isfinite(value) && value == long_hidden.row(row)[column],
                          "all-future flash final output preserves residual");
        }
    }

    flash_scratch.logits.swap(scratch.logits);
    std::fill(flash_scratch.logits.begin(),
              flash_scratch.logits.end(),
              123.0f);
    const std::vector<float> poisoned_logits(flash_scratch.logits);
    const float* logits_data = flash_scratch.logits.data();
    const size_t logits_size = flash_scratch.logits.size();
    const size_t logits_capacity = flash_scratch.logits.capacity();

    const ActivationBuffer short_hidden_again = make_attention_hidden(2);
    ActivationBuffer reference_short_output;
    run_attention(weights,
                  operators,
                  plan,
                  80,
                  reference_cache,
                  reference_scratch,
                  short_hidden_again,
                  reference_short_output,
                  reference_flags);
    ActivationBuffer accelerated_short_output;
    run_attention(weights,
                  operators,
                  plan,
                  80,
                  accelerated_cache,
                  flash_scratch,
                  short_hidden_again,
                  accelerated_short_output,
                  accelerated_flags);
    check_mxfp4_outputs(accelerated_short_output, reference_short_output, 1e-4f);
    scratch_check(flash_scratch.logits.data() == logits_data,
                  "flash reuse preserves prior logits storage");
    scratch_check(flash_scratch.logits.size() == logits_size,
                  "flash reuse preserves prior logits size");
    scratch_check(flash_scratch.logits.capacity() == logits_capacity,
                  "flash reuse preserves prior logits capacity");
    scratch_check(std::equal(flash_scratch.logits.begin(),
                             flash_scratch.logits.end(),
                             poisoned_logits.begin()),
                  "flash reuse leaves prior logits untouched");
    scratch_check(flash_scratch.workspace.data() == worker_data, "flash worker storage reuse");
    scratch_check(flash_scratch.workspace.capacity() == worker_capacity, "flash worker capacity reuse");

    if (attention_test_thread_count > 1)
    {
        constexpr uint64_t small_context = 576;
        LayerCache small_cache = make_float_attention_cache(split_plan.kv_head_count * split_plan.head_dimension,
                                                            small_context,
                                                            1024,
                                                            0);
        LayerCache small_reference_cache = small_cache;
        AttentionScratch small_scratch;
        AttentionScratch small_reference_scratch;
        const ActivationBuffer small_token = make_attention_hidden(1, split_plan.head_count * split_plan.head_dimension);
        ActivationBuffer small_output;
        ActivationBuffer small_reference_output;
        run_attention(weights,
                      operators,
                      split_plan,
                      small_context,
                      small_cache,
                      small_scratch,
                      small_token,
                      small_output,
                      accelerated_flags);
        run_attention(weights,
                      operators,
                      split_plan,
                      small_context,
                      small_reference_cache,
                      small_reference_scratch,
                      small_token,
                      small_reference_output,
                      reference_flags);
        check_mxfp4_outputs(small_output, small_reference_output, 1e-4f);
        scratch_check(small_scratch.flash_partial_max.empty(),
                      "small split workload falls back");

        constexpr uint64_t split_context = 1024;
        constexpr uint64_t split_capacity = 2048;
        const uint32_t split_columns = split_plan.kv_head_count * split_plan.head_dimension;
        const ActivationBuffer split_token = make_attention_hidden(1, split_plan.head_count * split_plan.head_dimension);
        LayerCache split_cache = make_float_attention_cache(split_columns,
                                                            split_context,
                                                            split_capacity,
                                                            0);
        LayerCache split_reference_cache = split_cache;
        AttentionScratch split_scratch;
        AttentionScratch split_reference_scratch;
        ActivationBuffer split_output;
        ActivationBuffer split_reference_output;
        run_attention(weights,
                      operators,
                      split_plan,
                      split_context,
                      split_cache,
                      split_scratch,
                      split_token,
                      split_output,
                      accelerated_flags);
        run_attention(weights,
                      operators,
                      split_plan,
                      split_context,
                      split_reference_cache,
                      split_reference_scratch,
                      split_token,
                      split_reference_output,
                      reference_flags);
        check_mxfp4_outputs(split_output, split_reference_output, 1e-4f);
        scratch_check(split_scratch.logits.empty(), "cold split-KV leaves logits untouched");
        scratch_check(split_scratch.logits.capacity() == 0, "cold split-KV does not allocate logits");
        scratch_check(!split_scratch.workspace.empty(), "split worker storage allocation");
        scratch_check(split_scratch.flash_partial_max.size()
                          == static_cast<size_t>(split_plan.head_count) * attention_test_thread_count,
                      "split partial maximum storage");
        const float* split_worker_data = split_scratch.workspace.data();
        const size_t split_worker_capacity = split_scratch.workspace.capacity();

        std::fill(split_scratch.workspace.begin(), split_scratch.workspace.end(),
                  std::numeric_limits<float>::quiet_NaN());
        std::fill(split_scratch.flash_partial_max.begin(), split_scratch.flash_partial_max.end(),
                  std::numeric_limits<float>::quiet_NaN());
        std::fill(split_scratch.flash_partial_sum.begin(), split_scratch.flash_partial_sum.end(),
                  std::numeric_limits<float>::quiet_NaN());
        std::fill(split_scratch.flash_partial_output.begin(), split_scratch.flash_partial_output.end(),
                  std::numeric_limits<float>::quiet_NaN());

        LayerCache ring_cache = make_float_attention_cache(split_columns,
                                                           split_context,
                                                           split_capacity,
                                                           split_capacity - 512);
        LayerCache ring_reference_cache = make_float_attention_cache(split_columns,
                                                                     split_context,
                                                                     split_capacity,
                                                                     0);
        AttentionScratch ring_reference_scratch;
        ActivationBuffer ring_output;
        ActivationBuffer ring_reference_output;
        run_attention(weights,
                      operators,
                      split_plan,
                      split_context,
                      ring_cache,
                      split_scratch,
                      split_token,
                      ring_output,
                      accelerated_flags);
        run_attention(weights,
                      operators,
                      split_plan,
                      split_context,
                      ring_reference_cache,
                      ring_reference_scratch,
                      split_token,
                      ring_reference_output,
                      reference_flags);
        check_mxfp4_outputs(ring_output, ring_reference_output, 1e-4f);
        scratch_check(split_scratch.workspace.data() == split_worker_data,
                      "split worker storage reuse");
        scratch_check(split_scratch.workspace.capacity() == split_worker_capacity,
                      "split worker capacity reuse");

        thread_limit_scope.set(1);
        LayerCache single_thread_cache = make_float_attention_cache(split_columns,
                                                                    split_context,
                                                                    split_capacity,
                                                                    0);
        ActivationBuffer single_thread_output;
        run_attention(weights,
                      operators,
                      split_plan,
                      split_context,
                      single_thread_cache,
                      split_scratch,
                      split_token,
                      single_thread_output,
                      accelerated_flags);
        check_mxfp4_outputs(single_thread_output, split_reference_output, 1e-4f);

        thread_limit_scope.set(attention_test_thread_count);
        LayerCache split_again_cache = make_float_attention_cache(split_columns,
                                                                  split_context,
                                                                  split_capacity,
                                                                  0);
        ActivationBuffer split_again_output;
        run_attention(weights,
                      operators,
                      split_plan,
                      split_context,
                      split_again_cache,
                      split_scratch,
                      split_token,
                      split_again_output,
                      accelerated_flags);
        check_mxfp4_outputs(split_again_output, split_reference_output, 1e-4f);

        constexpr uint64_t multithread_context = 4096;
        const LayerCache initial_multithread_cache = make_float_attention_cache(split_columns,
                                                                                multithread_context,
                                                                                multithread_context * 2,
                                                                                0);
        const uint64_t multithread_operation_count = static_cast<uint64_t>(split_plan.head_count)
                                                     * split_plan.head_dimension
                                                     * (multithread_context + 1);
        const int expected_head_threads = cpu_linear_team_size(2 * multithread_operation_count,
                                                               DType::Float32);
        scratch_check(expected_head_threads > 1, "multithread logits fixture selects head workers");

        LayerCache multithread_cache = initial_multithread_cache;
        AttentionScratch multithread_scratch;
        ActivationBuffer multithread_logits_output;
        run_attention(weights,
                      operators,
                      split_plan,
                      multithread_context,
                      multithread_cache,
                      multithread_scratch,
                      split_token,
                      multithread_logits_output,
                      reference_flags);
        const size_t expected_logits_size = static_cast<size_t>(multithread_context + 1)
                                            * expected_head_threads;
        scratch_check(multithread_scratch.logits.size() == expected_logits_size,
                      "head workers use the logical logits stride");
        const float* logits_data = multithread_scratch.logits.data();
        const size_t logits_capacity = multithread_scratch.logits.capacity();
        std::fill(multithread_scratch.logits.begin(),
                  multithread_scratch.logits.end(),
                  std::numeric_limits<float>::quiet_NaN());

        LayerCache poisoned_cache = initial_multithread_cache;
        ActivationBuffer poisoned_logits_output;
        run_attention(weights,
                      operators,
                      split_plan,
                      multithread_context,
                      poisoned_cache,
                      multithread_scratch,
                      split_token,
                      poisoned_logits_output,
                      reference_flags);
        check_mxfp4_outputs(poisoned_logits_output, multithread_logits_output, 1e-4f);
        scratch_check(multithread_scratch.logits.data() == logits_data,
                      "head logits storage reuse");
        scratch_check(multithread_scratch.logits.capacity() == logits_capacity,
                      "head logits capacity reuse");
    }
}

void test_reusable_kernel_scratch()
{
    test_mxfp4_scratch_reuse();
    test_attention_scratch_reuse();
}

} // namespace moe
} // namespace ncnn
