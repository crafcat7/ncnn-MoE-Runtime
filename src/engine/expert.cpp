#include "expert.h"

#include "sessionstate.h"
#include "expertbackend.h"
#include "graph/compiledmodel.h"
#include "kernels/bfloat16.h"
#include "kernels/fastmath.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"
#include "storage/expertcache.h"
#include "storage/weightstore.h"
#include "backends/ncnn/linear.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <span>
#include <utility>

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

static constexpr size_t expert_prefetch_limit_size = 4 * 1024;
static constexpr size_t cache_line_size = 64;

static void prefetch_address(const void* address)
{
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(address, 0, 3);
#else
    (void)address;
#endif
}

static uint64_t prefetch_buffer(const void* data, size_t size)
{
    if (!data || size == 0)
        return 0;
    const size_t hinted_size = std::min(size, expert_prefetch_limit_size);
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    for (size_t offset = 0; offset < hinted_size; offset += cache_line_size)
        prefetch_address(ptr + offset);
    return hinted_size;
}

static uint64_t prefetch_tensor(const TensorData& tensor)
{
    if (tensor.dtype == DType::Float32)
    {
        const std::span<const float> values = tensor.float32_values();
        return prefetch_buffer(values.data(), values.size() * sizeof(float));
    }
    if (tensor.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> values = tensor.bfloat16_values();
        return prefetch_buffer(values.data(), values.size() * sizeof(uint16_t));
    }
    if (tensor.dtype == DType::Int8)
    {
        const std::span<const int8_t> values = tensor.int8_values();
        return prefetch_buffer(values.data(), values.size());
    }
    if (tensor.dtype == DType::Float8E4M3)
    {
        const std::span<const uint8_t> values = tensor.float8_values();
        return prefetch_buffer(values.data(), values.size());
    }
    if (tensor.dtype == DType::Int64)
    {
        const std::span<const int64_t> values = tensor.int64_values();
        return prefetch_buffer(values.data(), values.size() * sizeof(int64_t));
    }
    if (tensor.dtype == DType::MxFp4)
    {
        return prefetch_buffer(tensor.mxfp4_blocks.data(), tensor.mxfp4_blocks.size()) + prefetch_buffer(tensor.mxfp4_scales.data(), tensor.mxfp4_scales.size());
    }
    if (is_qnk_dtype(tensor.dtype))
    {
        const std::span<const uint8_t> values = tensor.qnk_values();
        return prefetch_buffer(values.data(), values.size());
    }
    return 0;
}

static uint64_t prefetch_weight(const WeightStore& weights, TensorHandle handle)
{
    return handle == invalid_tensor_handle ? 0 : prefetch_tensor(weights.at(handle));
}

static float activate(
    float value,
    ExpertActivation activation,
    float limit,
    uint64_t optimization_flags)
{
    switch (activation)
    {
    case ExpertActivation::Relu: return std::max(0.0f, value);
    case ExpertActivation::Silu:
        return scaled_silu(value, 1.0f, optimization_flags);
    case ExpertActivation::Gelu: return 0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
    case ExpertActivation::ClampedSilu:
    {
        const float clamped = limit > 0.0f ? std::clamp(value, -limit, limit) : value;
        return scaled_silu(clamped, 1.0f, optimization_flags);
    }
    case ExpertActivation::DeepSeekSwiGlu:
    {
        const float clamped = limit > 0.0f ? std::min(value, limit) : value;
        return scaled_silu(clamped, 1.0f, optimization_flags);
    }
    case ExpertActivation::GptOssSwiGlu: return value;
    }
    return value;
}

void record_mxfp4(const TensorData& matrix, size_t input_rows, ExpertExecutionMetrics& metrics)
{
    if (matrix.dtype != DType::MxFp4)
        return;
    const uint64_t rows = static_cast<uint64_t>(matrix.shape[0]) * input_rows;
    metrics.mxfp4_paired_rows += static_cast<uint64_t>(matrix.shape[0] / 2) * 2 * input_rows;
    if (input_rows == 1)
        metrics.mxfp4_decode_gemv_rows += rows;
    else
        metrics.mxfp4_prefill_gemm_rows += rows;
}

static CpuBatch expert_linear(const TensorData& matrix, const TensorData* bias, const CompiledOperator* executable, const CpuBatch& input, ExpertExecutionMetrics& metrics, uint64_t optimization_flags)
{
    record_mxfp4(matrix, input.rows(), metrics);
    return bias ? linear_batch(matrix, *bias, input, optimization_flags, executable) : linear_batch(matrix, input, optimization_flags, executable);
}

static CpuBatch run_expert(const WeightStore& weights, const CompiledOperatorTable& operators, const ExpertPlan& expert, const ExpertCacheLease* cached_weights, const CpuBatch& input, bool prefetch, ExpertExecutionMetrics& metrics, uint64_t optimization_flags)
{
    if (expert.gate_up_weight != invalid_tensor_handle)
    {
        const TensorData& gate_up_weight = cached_weights && cached_weights->gate_up ? *cached_weights->gate_up : weights.at(expert.gate_up_weight);
        const CompiledOperator* gate_up_operator = cached_weights && cached_weights->gate_up
                                                       ? cached_weights->gate_up_operator
                                                       : operators.find_weight(expert.gate_up_weight);
        if (prefetch)
            metrics.hinted_bytes += prefetch_tensor(gate_up_weight);
        const TensorData* gate_up_bias = expert.gate_up_bias == invalid_tensor_handle ? nullptr : &weights.at(expert.gate_up_bias);
        CpuBatch activated;
        if (gate_up_weight.dtype == DType::MxFp4)
        {
            activated = fused_mxfp4_gate_up_batch(
                gate_up_weight,
                gate_up_bias,
                input,
                expert.activation,
                expert.activation_limit,
                optimization_flags);
            metrics.mxfp4_fused_gate_up_rows += static_cast<uint64_t>(activated.rows()) * activated.columns();
            record_mxfp4(gate_up_weight, input.rows(), metrics);
        }
        else if (expert.layout == ExpertLayout::PackedGateUpDown)
        {
            CpuBatch gate_up = expert_linear(gate_up_weight, gate_up_bias, gate_up_operator, input, metrics, optimization_flags);
            activated = CpuBatch(gate_up.rows(), gate_up.columns() / 2);
            for (size_t token_index = 0; token_index < gate_up.rows(); ++token_index)
            {
                const float* source = gate_up.row(token_index);
                float* destination = activated.row(token_index);
                for (uint32_t column = 0; column < activated.columns(); ++column)
                {
                    const float gate = source[column];
                    const float up = source[activated.columns() + column];
                    destination[column] = activate(
                                              gate,
                                              expert.activation,
                                              expert.activation_limit,
                                              optimization_flags)
                                          * up;
                }
            }
        }
        else
        {
            CpuBatch gate_up = expert_linear(gate_up_weight, gate_up_bias, gate_up_operator, input, metrics, optimization_flags);
            activated = CpuBatch(gate_up.rows(), gate_up.columns() / 2);
            for (size_t token_index = 0; token_index < gate_up.rows(); ++token_index)
            {
                const float* source = gate_up.row(token_index);
                float* destination = activated.row(token_index);
                for (uint32_t column = 0; column < activated.columns(); ++column)
                {
                    const float gate = expert.activation_limit > 0.0f ? std::min(source[column * 2], expert.activation_limit) : source[column * 2];
                    const float linear = expert.activation_limit > 0.0f
                                             ? std::clamp(source[column * 2 + 1], -expert.activation_limit, expert.activation_limit)
                                             : source[column * 2 + 1];
                    const float silu = scaled_silu(
                        gate,
                        1.702f,
                        optimization_flags);
                    destination[column] = silu * (linear + 1.0f);
                }
            }
        }
        const TensorData& down_weight = cached_weights && cached_weights->down ? *cached_weights->down : weights.at(expert.down_weight);
        const CompiledOperator* down_operator = cached_weights && cached_weights->down
                                                    ? cached_weights->down_operator
                                                    : operators.find_weight(expert.down_weight);
        if (prefetch)
            metrics.hinted_bytes += prefetch_tensor(down_weight);
        return expert_linear(down_weight, expert.down_bias == invalid_tensor_handle ? nullptr : &weights.at(expert.down_bias), down_operator, activated, metrics, optimization_flags);
    }

    if (prefetch)
        metrics.hinted_bytes += prefetch_weight(weights, expert.up_weight);
    const TensorData& up_weight = weights.at(expert.up_weight);
    bool gate_prefetched = false;
    if (expert.layout != ExpertLayout::UpDown && expert.gate_weight != invalid_tensor_handle)
    {
        const TensorData& gate_weight = weights.at(expert.gate_weight);
        const TensorData& down_weight = weights.at(expert.down_weight);
        const CompiledOperator* gate_operator = operators.find_weight(expert.gate_weight);
        const CompiledOperator* up_operator = operators.find_weight(expert.up_weight);
        const CompiledOperator* down_operator = operators.find_weight(expert.down_weight);
        if (has_flag(optimization_flags, OptimizationCpuFloat8FusedGateUp)
            && gate_weight.dtype == DType::Float8E4M3
            && up_weight.dtype == DType::Float8E4M3)
        {
            if (prefetch)
            {
                metrics.hinted_bytes += prefetch_weight(weights, expert.gate_weight);
                gate_prefetched = true;
            }
            CpuBatch activated;
            if (fused_float8_gate_up_batch(
                    gate_weight,
                    up_weight,
                    input,
                    expert.activation,
                    expert.activation_limit,
                    activated,
                    optimization_flags,
                    gate_operator,
                    up_operator))
            {
                if (prefetch)
                {
                    metrics.hinted_bytes += prefetch_weight(weights, expert.down_weight);
                }
                return expert_linear(down_weight, nullptr, down_operator, activated, metrics, optimization_flags);
            }
        }
    }
    CpuBatch up = expert_linear(up_weight, nullptr, operators.find_weight(expert.up_weight), input, metrics, optimization_flags);
    if (expert.layout != ExpertLayout::UpDown)
    {
        if (prefetch && !gate_prefetched)
            metrics.hinted_bytes += prefetch_weight(weights, expert.gate_weight);
        const CpuBatch gate = expert_linear(weights.at(expert.gate_weight), nullptr, operators.find_weight(expert.gate_weight), input, metrics, optimization_flags);
        for (size_t token_index = 0; token_index < up.rows(); ++token_index)
        {
            float* up_row = up.row(token_index);
            const float* gate_row = gate.row(token_index);
            for (uint32_t column = 0; column < up.columns(); ++column)
            {
                if (expert.activation == ExpertActivation::DeepSeekSwiGlu && expert.activation_limit > 0.0f)
                    up_row[column] = std::clamp(up_row[column], -expert.activation_limit, expert.activation_limit);
                up_row[column] *= activate(
                    gate_row[column],
                    expert.activation,
                    expert.activation_limit,
                    optimization_flags);
            }
        }
    }
    else
    {
        for (size_t token_index = 0; token_index < up.rows(); ++token_index)
        {
            float* token = up.row(token_index);
            for (uint32_t column = 0; column < up.columns(); ++column)
                token[column] = activate(
                    token[column],
                    expert.activation,
                    expert.activation_limit,
                    optimization_flags);
        }
    }
    if (prefetch)
        metrics.hinted_bytes += prefetch_weight(weights, expert.down_weight);
    return expert_linear(weights.at(expert.down_weight), nullptr, operators.find_weight(expert.down_weight), up, metrics, optimization_flags);
}

CpuBatch forward_shared_expert(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    const CpuBatch& input,
    ExpertExecutionMetrics& metrics,
    uint64_t optimization_flags)
{
    const ExpertPlan& expert = moe.shared_expert;
    const bool has_router_gate = moe.shared_expert_gate_weight != invalid_tensor_handle;
    const CompiledOperator& fused_shared_operator = model.operators.at(moe.fused_shared_input_bfloat16_operator);
    if (fused_shared_operator.bfloat16
        && expert.down_weight != invalid_tensor_handle)
    {
        const CompiledOperator& down_operator = model.operators.at_weight(expert.down_weight);
        if (down_operator.bfloat16)
        {
            const uint32_t intermediate = model.weights.at(expert.up_weight).shape[0];
            CpuBatch output;
            if (fused_shared_operator.bfloat16->forward_swiglu_chain(
                    input,
                    *down_operator.bfloat16,
                    intermediate,
                    expert.activation,
                    expert.activation_limit,
                    has_router_gate,
                    output))
            {
                return output;
            }
        }
    }
    if (fused_shared_operator.bfloat16)
    {
        CpuBatch fused;
        if (fused_shared_operator.bfloat16->forward(
                input,
                fused))
        {
            const uint32_t intermediate = model.weights.at(expert.up_weight).shape[0];
            const uint32_t expected_columns = intermediate * 2 + (has_router_gate ? 1 : 0);
            if (fused.columns() == expected_columns)
            {
                CpuBatch activated(
                    fused.rows(),
                    intermediate);
                for (size_t token_index = 0;
                     token_index < fused.rows();
                     ++token_index)
                {
                    const float* source = fused.row(token_index);
                    float* destination = activated.row(token_index);
                    for (uint32_t column = 0;
                         column < intermediate;
                         ++column)
                    {
                        destination[column] = activate(
                                                  source[column],
                                                  expert.activation,
                                                  expert.activation_limit,
                                                  optimization_flags)
                                              * source[intermediate + column];
                    }
                }
                CpuBatch output = expert_linear(
                    model.weights.at(expert.down_weight),
                    nullptr,
                    model.operators.find_weight(expert.down_weight),
                    activated,
                    metrics,
                    optimization_flags);
                if (has_router_gate)
                {
                    for (size_t token_index = 0;
                         token_index < output.rows();
                         ++token_index)
                    {
                        const float scale = 1.0f
                                            / (1.0f
                                               + float_approximate_exp(
                                                   -fused.row(token_index)
                                                        [intermediate * 2]));
                        float* token = output.row(token_index);
                        for (uint32_t column = 0;
                             column < output.columns();
                             ++column)
                        {
                            token[column] *= scale;
                        }
                    }
                }
                return output;
            }
        }
    }

    CpuBatch output = run_expert(
        model.weights,
        model.operators,
        moe.shared_expert,
        nullptr,
        input,
        false,
        metrics,
        optimization_flags);
    if (moe.shared_expert_gate_weight == invalid_tensor_handle)
        return output;
    CpuBatch gate = linear_batch(
        model.weights.at(moe.shared_expert_gate_weight),
        input,
        optimization_flags,
        model.operators.find_weight(moe.shared_expert_gate_weight));
    assert(gate.columns() == 1);
    for (size_t token_index = 0; token_index < output.rows(); ++token_index)
    {
        const float scale = 1.0f / (1.0f + float_approximate_exp(-gate.row(token_index)[0]));
        float* token = output.row(token_index);
        for (uint32_t column = 0; column < output.columns(); ++column)
            token[column] *= scale;
    }
    return output;
}

static uint64_t elapsed_microseconds(std::chrono::steady_clock::time_point start)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

static void gather_tokens(const CpuBatch& source, const std::vector<ExpertRoute>& routes, CpuBatch& gathered)
{
    gathered.reset(routes.size(), source.columns(), false);
    for (size_t route_index = 0; route_index < routes.size(); ++route_index)
    {
        std::copy_n(source.row(routes[route_index].token_index), source.columns(), gathered.row(route_index));
    }
}

struct HybridBlockState
{
    size_t active_index = 0;
    size_t input_begin = 0;
    CpuBatch input;
    CpuBatch output;
    bool gpu_planned = false;
    bool gpu_executed = false;
    bool gpu_aggregated = false;
    bool cpu_executed = false;
};

struct HybridCpuBlockGroup
{
    size_t active_index = 0;
    std::vector<size_t> block_indices;
    CpuBatch input;
    bool executed = false;
};

static size_t hybrid_block_end(
    std::span<const ExpertRoute> routes,
    size_t begin,
    uint32_t token_block_size)
{
    if (begin >= routes.size())
        return begin;

    const uint32_t first_token = routes[begin].token_index;
    const uint64_t token_limit = static_cast<uint64_t>(first_token) + token_block_size;
    size_t end = begin;
    while (end < routes.size()
           && static_cast<uint64_t>(routes[end].token_index) < token_limit)
    {
        ++end;
    }
    return end;
}

static uint64_t run_hybrid_cpu_blocks(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    LayerGraphState& layer_state,
    std::vector<HybridBlockState>& blocks,
    std::span<const uint8_t> active_ready,
    bool prefetch)
{
    const size_t invalid_group = std::numeric_limits<size_t>::max();
    std::vector<size_t> group_for_active(active_ready.size(), invalid_group);
    std::vector<HybridCpuBlockGroup> groups;
    groups.reserve(blocks.size());
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    {
        const HybridBlockState& block = blocks[block_index];
        if (block.active_index < active_ready.size()
            && active_ready[block.active_index] == 1
            && !block.gpu_planned
            && !block.cpu_executed)
        {
            size_t& group_index = group_for_active[block.active_index];
            if (group_index == invalid_group)
            {
                group_index = groups.size();
                groups.push_back({});
                groups.back().active_index = block.active_index;
            }
            groups[group_index].block_indices.push_back(block_index);
        }
    }
    if (groups.empty())
        return 0;

    // Gather multiple CPU blocks per Expert for the batched kernel.
    for (HybridCpuBlockGroup& group : groups)
    {
        if (group.block_indices.size() <= 1)
            continue;
        const HybridBlockState& first = blocks[group.block_indices.front()];
        size_t total_rows = 0;
        for (size_t block_index : group.block_indices)
            total_rows += blocks[block_index].input.rows();
        group.input.reset(total_rows, first.input.columns(), false);
        size_t row_offset = 0;
        for (size_t block_index : group.block_indices)
        {
            const CpuBatch& input = blocks[block_index].input;
            for (size_t row = 0; row < input.rows(); ++row)
            {
                std::copy_n(
                    input.row(row),
                    input.columns(),
                    group.input.row(row_offset + row));
            }
            row_offset += input.rows();
        }
    }

    const auto compute_start = std::chrono::steady_clock::now();
    int team_size = 1;
    bool parallelize = false;
#if defined(_OPENMP)
    team_size = std::min(static_cast<int>(groups.size()), static_cast<int>(cpu_linear_num_threads()));
    parallelize = team_size > 1;
#endif
    Bfloat16BatchedLinearExecutionCounter* const bfloat16_counter = current_bfloat16_batched_linear_execution_counter();
    const int64_t group_count = static_cast<int64_t>(groups.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(team_size) if (parallelize)
    for (int64_t group_index = 0; group_index < group_count; ++group_index)
    {
        HybridCpuBlockGroup& group = groups[static_cast<size_t>(group_index)];
        const ScopedBfloat16BatchedLinearExecutionCounter bfloat16_scope(
            bfloat16_counter);
        ActiveExpertExecution& active = layer_state.active_experts[group.active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        if (group.block_indices.size() == 1)
        {
            HybridBlockState& block = blocks[group.block_indices.front()];
            block.output = run_expert(
                model.weights,
                model.operators,
                expert,
                active.lease.gate_up ? &active.lease : nullptr,
                block.input,
                prefetch,
                active.metrics,
                model.opt.optimization_flags);
            block.cpu_executed = block.output.rows() == block.input.rows()
                                 && block.output.columns() != 0;
            group.executed = block.cpu_executed;
            continue;
        }

        CpuBatch group_output = run_expert(
            model.weights,
            model.operators,
            expert,
            active.lease.gate_up ? &active.lease : nullptr,
            group.input,
            prefetch,
            active.metrics,
            model.opt.optimization_flags);
        if (group_output.rows() != group.input.rows()
            || group_output.columns() == 0)
        {
            continue;
        }
        size_t row_offset = 0;
        group.executed = true;
        for (size_t block_index : group.block_indices)
        {
            HybridBlockState& block = blocks[block_index];
            block.output.reset(block.input.rows(), group_output.columns(), false);
            for (size_t row = 0; row < block.input.rows(); ++row)
            {
                std::copy_n(
                    group_output.row(row_offset + row),
                    group_output.columns(),
                    block.output.row(row));
            }
            block.cpu_executed = block.output.rows() == block.input.rows();
            row_offset += block.input.rows();
        }
    }

    const uint64_t elapsed = elapsed_microseconds(compute_start);
    if (model.expert_backend)
    {
        const uint64_t share = std::max<uint64_t>(1, elapsed / groups.size());
        for (const HybridCpuBlockGroup& group : groups)
        {
            if (!group.executed)
                continue;
            const ActiveExpertExecution& active = layer_state.active_experts[group.active_index];
            uint64_t group_rows = 0;
            for (size_t block_index : group.block_indices)
                group_rows += blocks[block_index].input.rows();
            model.expert_backend->observe_cpu(
                static_cast<uint32_t>(group_rows),
                moe.experts[active.batch.expert_id].weight_size,
                share);
        }
    }
    return elapsed;
}

static void assemble_hybrid_outputs(
    LayerGraphState& layer_state,
    std::span<const HybridBlockState> blocks,
    std::span<const uint8_t> backend_aggregated)
{
    std::vector<uint8_t> initialized(layer_state.active_experts.size(), 0);
    for (const HybridBlockState& block : blocks)
    {
        if (block.active_index >= layer_state.active_experts.size()
            || block.active_index >= backend_aggregated.size()
            || block.gpu_aggregated
            || !block.cpu_executed && !block.gpu_executed)
        {
            continue;
        }

        ActiveExpertExecution& active = layer_state.active_experts[block.active_index];
        if (initialized[block.active_index] == 0)
        {
            active.output.reset(
                active.input.rows(),
                block.output.columns(),
                true);
            initialized[block.active_index] = 1;
        }
        if (block.output.rows() != block.input.rows()
            || block.output.columns() != active.output.columns()
            || block.input_begin + block.output.rows() > active.output.rows())
        {
            continue;
        }
        for (size_t row = 0; row < block.output.rows(); ++row)
        {
            std::copy_n(
                block.output.row(row),
                block.output.columns(),
                active.output.row(block.input_begin + row));
        }
    }
}

static uint64_t run_experts(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    LayerGraphState& layer_state,
    std::span<const size_t> active_indices,
    CpuExpertExecutionScratch& scratch,
    bool prefetch)
{
    if (active_indices.empty())
        return 0;

    const auto compute_start = std::chrono::steady_clock::now();
    std::vector<Mxfp4Task>& decode_tasks = scratch.decode_tasks;
    decode_tasks.clear();
    decode_tasks.reserve(active_indices.size());
    for (size_t active_index : active_indices)
    {
        ActiveExpertExecution& active = layer_state.active_experts[active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        Mxfp4Task task;
        if (active.lease.gate_up)
        {
            task.gate_up = active.lease.gate_up.get();
            task.gate_up_operator = active.lease.gate_up_operator;
        }
        else if (expert.gate_up_weight != invalid_tensor_handle)
        {
            task.gate_up = &model.weights.at(expert.gate_up_weight);
            task.gate_up_operator = model.operators.find_weight(expert.gate_up_weight);
        }
        task.gate_up_bias = expert.gate_up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_bias);
        if (active.lease.down)
        {
            task.down = active.lease.down.get();
            task.down_operator = active.lease.down_operator;
        }
        else if (expert.down_weight != invalid_tensor_handle)
        {
            task.down = &model.weights.at(expert.down_weight);
            task.down_operator = model.operators.find_weight(expert.down_weight);
        }
        task.down_bias = expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias);
        task.input = &active.input;
        task.output = &active.output;
        task.activation = expert.activation;
        task.activation_limit = expert.activation_limit;
        decode_tasks.push_back(task);
    }

    const bool prefetched_batch_weights = prefetch
                                          && std::all_of(
                                              decode_tasks.begin(),
                                              decode_tasks.end(),
                                              [](const Mxfp4Task& task) {
                                                  return task.gate_up
                                                         && task.down
                                                         && task.gate_up->dtype == DType::MxFp4
                                                         && task.down->dtype == DType::MxFp4;
                                              });
    if (prefetched_batch_weights)
    {
        for (size_t task_index = 0; task_index < decode_tasks.size(); ++task_index)
        {
            ActiveExpertExecution& active = layer_state.active_experts[active_indices[task_index]];
            active.metrics.hinted_bytes += prefetch_tensor(*decode_tasks[task_index].gate_up);
            active.metrics.hinted_bytes += prefetch_tensor(*decode_tasks[task_index].down);
        }
    }

    const bool grouped_decode = mxfp4_expert_batch(decode_tasks, &scratch.kernels, model.opt.optimization_flags);
    if (grouped_decode)
    {
        for (size_t task_index = 0; task_index < active_indices.size(); ++task_index)
        {
            ActiveExpertExecution& active = layer_state.active_experts[active_indices[task_index]];
            record_mxfp4(*decode_tasks[task_index].gate_up, active.input.rows(), active.metrics);
            record_mxfp4(*decode_tasks[task_index].down, active.input.rows(), active.metrics);
            active.metrics.mxfp4_fused_gate_up_rows += static_cast<uint64_t>(active.input.rows())
                                                       * decode_tasks[task_index].gate_up->shape[0] / 2;
            if (task_index
                < scratch.kernels.physical_input_rows.size())
            {
                active.metrics.mxfp4_reused_input_rows += active.input.rows()
                                                          - scratch.kernels.physical_input_rows[task_index];
            }
        }
        const uint64_t elapsed = elapsed_microseconds(compute_start);
        if (model.expert_backend)
        {
            const uint64_t share = std::max<uint64_t>(1, elapsed / active_indices.size());
            for (size_t active_index : active_indices)
            {
                const ActiveExpertExecution& active = layer_state.active_experts[active_index];
                const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                model.expert_backend->observe_cpu(static_cast<uint32_t>(active.input.rows()), expert.weight_size, share);
            }
        }
        return elapsed;
    }

    bool parallelize_experts = false;
    int expert_team_size = 1;
#if defined(_OPENMP)
    expert_team_size = std::min(static_cast<int>(active_indices.size()), static_cast<int>(cpu_linear_num_threads()));
    parallelize_experts = expert_team_size > 1;
#endif
    const int64_t parallel_expert_count = static_cast<int64_t>(active_indices.size());
    Bfloat16BatchedLinearExecutionCounter* const bfloat16_counter = current_bfloat16_batched_linear_execution_counter();
#pragma omp parallel for schedule(dynamic, 1) num_threads(expert_team_size) if (parallelize_experts)
    for (int64_t task_index = 0; task_index < parallel_expert_count; ++task_index)
    {
        const ScopedBfloat16BatchedLinearExecutionCounter bfloat16_scope(
            bfloat16_counter);
        ActiveExpertExecution& active = layer_state.active_experts[active_indices[static_cast<size_t>(task_index)]];
        const uint32_t expert_id = active.batch.expert_id;
        active.output = run_expert(model.weights, model.operators, moe.experts[expert_id], active.lease.gate_up ? &active.lease : nullptr, active.input, prefetch && !prefetched_batch_weights, active.metrics, model.opt.optimization_flags);
    }
    const uint64_t elapsed = elapsed_microseconds(compute_start);
    if (model.expert_backend)
    {
        const uint64_t share = std::max<uint64_t>(1, elapsed / active_indices.size());
        for (size_t active_index : active_indices)
        {
            const ActiveExpertExecution& active = layer_state.active_experts[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            model.expert_backend->observe_cpu(static_cast<uint32_t>(active.input.rows()), expert.weight_size, share);
        }
    }
    return elapsed;
}

bool can_run_vulkan_expert(
    const ExpertPlan& expert,
    const TensorData& gate_up,
    const TensorData& down,
    uint64_t optimization_flags)
{
    if ((expert.activation != ExpertActivation::Silu
         && expert.activation != ExpertActivation::GptOssSwiGlu
         && expert.activation != ExpertActivation::DeepSeekSwiGlu)
        || gate_up.shape.size() != 2
        || down.shape.size() != 2
        || gate_up.shape[0] % 2 != 0
        || down.shape[1] != gate_up.shape[0] / 2)
    {
        return false;
    }
    if (gate_up.dtype == DType::MxFp4 && down.dtype == DType::MxFp4)
        return true;
    if (gate_up.dtype == DType::BFloat16 && down.dtype == DType::BFloat16)
    {
        return expert.activation == ExpertActivation::Silu
               && gate_up.shape[0] / 2 % 128 == 0
               && gate_up.bfloat16_values().size() == gate_up.element_count()
               && down.bfloat16_values().size() == down.element_count();
    }
    return has_flag(optimization_flags, OptimizationVulkanQnK)
           && is_qnk_dtype(gate_up.dtype)
           && gate_up.dtype == down.dtype
           && qnk_shape_supported(gate_up.dtype, gate_up.shape[0], gate_up.shape[1])
           && qnk_shape_supported(down.dtype, down.shape[0], down.shape[1]);
}

// Resident Qn_K weights are owned by CompiledModel::weights.  The Vulkan
// backend only needs a temporary shared handle while its asynchronous
// admission worker copies the bytes to device storage; the model outlives
// the backend, so this non-owning handle avoids copying a large raw tensor.
static std::shared_ptr<const TensorData> borrow_resident_tensor(const TensorData& tensor)
{
    return std::shared_ptr<const TensorData>(&tensor, [](const TensorData*) {});
}

static void admit_vulkan_expert(
    const CompiledModel& model,
    const ExpertPlan& expert,
    const ExpertCacheLease& lease,
    uint32_t residency_group,
    uint32_t token_count,
    ExecutionBackend backend)
{
    if (backend != ExecutionBackend::Vulkan
        || !model.expert_backend
        || !lease.gate_up
        || !lease.down
        || !can_run_vulkan_expert(expert, *lease.gate_up, *lease.down, model.opt.optimization_flags))
    {
        return;
    }
    const bool bfloat16_expert = lease.gate_up->dtype == DType::BFloat16
                                 && lease.down->dtype == DType::BFloat16;
    if (model.opt.hybrid_mode != HybridMode::HybridExperts
        && token_count < (bfloat16_expert
                              ? vulkan_expert_gpu_admission_min_rows
                              : vulkan_expert_gpu_victim_min_rows))
        return;
    model.expert_backend->admit(expert.cache_key, lease.gate_up, expert.gate_up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_bias), lease.down,
                                expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias), residency_group,
                                expert.activation_limit, expert.activation);
}

ExpertVictimExecutionMetadata victim_metadata(
    const CompiledModel& model,
    const ExpertPlan& expert,
    size_t token_count)
{
    ExpertVictimExecutionMetadata metadata;
    if (token_count < vulkan_expert_gpu_victim_min_rows
        && model.opt.hybrid_mode != HybridMode::HybridExperts)
        return metadata;
    if (expert.gate_up_weight == invalid_tensor_handle
        || expert.down_weight == invalid_tensor_handle
        || model.weights.at(expert.gate_up_weight).dtype != DType::MxFp4
        || model.weights.at(expert.down_weight).dtype != DType::MxFp4
        || !can_run_vulkan_expert(
            expert,
            model.weights.at(expert.gate_up_weight),
            model.weights.at(expert.down_weight),
            model.opt.optimization_flags))
    {
        return metadata;
    }
    metadata.gate_up_bias = expert.gate_up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_bias);
    metadata.down_bias = expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias);
    metadata.activation_limit = expert.activation_limit;
    metadata.activation = expert.activation;
    metadata.enabled = true;
    return metadata;
}

// Use block scheduling only for sufficiently large prefill waves.
static bool should_use_hybrid_expert_blocks(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    const LayerGraphState& layer_state,
    ExecutionBackend backend) noexcept
{
    if (backend != ExecutionBackend::Vulkan || !model.expert_backend
        || layer_state.normalized.rows() < 32
        || layer_state.active_experts.empty())
    {
        return false;
    }

    size_t routed_rows = 0;
    for (const ActiveExpertExecution& active : layer_state.active_experts)
    {
        if (active.batch.routes.empty()
            || active.batch.expert_id >= moe.experts.size())
        {
            return false;
        }
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        if (expert.gate_up_weight == invalid_tensor_handle
            || expert.down_weight == invalid_tensor_handle
            || !can_run_vulkan_expert(
                expert,
                model.weights.at(expert.gate_up_weight),
                model.weights.at(expert.down_weight),
                model.opt.optimization_flags))
        {
            return false;
        }
        // BF16 Experts use the per-layer batched path below. The block
        // scheduler is tuned for MXFP4's indexed/packed kernels; splitting a
        // BF16 wave into 32-token blocks creates many tiny submissions and
        // repeats host/device staging without increasing arithmetic
        // parallelism.
        if (model.weights.at(expert.gate_up_weight).dtype == DType::BFloat16)
            return false;
        routed_rows += active.batch.routes.size();
    }
    return routed_rows >= 32;
}

static Result<void> run_hybrid_expert_blocks(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    LayerGraphState& layer_state,
    SessionStatistics& statistics,
    CpuExpertExecutionScratch& scratch,
    uint32_t residency_group,
    bool prefetch)
{
    static constexpr uint32_t prefill_token_block_size = 32;
    const size_t active_expert_count = layer_state.active_experts.size();
    const auto cache_management_start = std::chrono::steady_clock::now();
    const auto compute_start = std::chrono::steady_clock::now();

    for (ActiveExpertExecution& active : layer_state.active_experts)
        gather_tokens(layer_state.normalized, active.batch.routes, active.input);

    size_t total_routes = 0;
    for (const ActiveExpertExecution& active : layer_state.active_experts)
        total_routes += active.batch.routes.size();

    std::vector<HybridBlockState> blocks;
    blocks.reserve(total_routes);
    scratch.backend_aggregated.assign(active_expert_count, 0);
    scratch.backend_aggregated_output_valid = false;
    scratch.backend_aggregated_output.reset(
        layer_state.normalized.rows(),
        model.descriptor.hidden_size,
        true);
    std::vector<ExpertBackendRequest> requests;
    requests.reserve(total_routes);
    std::vector<uint8_t> block_aggregated;
    std::vector<uint8_t> cpu_ready(active_expert_count, 0);
    std::vector<size_t> cpu_active_indices;
    cpu_active_indices.reserve(active_expert_count);

    for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
    {
        ActiveExpertExecution& active = layer_state.active_experts[active_index];
        size_t route_begin = 0;
        while (route_begin < active.batch.routes.size())
        {
            const size_t route_end = hybrid_block_end(
                active.batch.routes,
                route_begin,
                prefill_token_block_size);
            HybridBlockState block;
            block.active_index = active_index;
            block.input_begin = route_begin;
            block.input.reset(route_end - route_begin, active.input.columns(), false);
            for (size_t row = route_begin; row < route_end; ++row)
            {
                std::copy_n(
                    active.input.row(row),
                    active.input.columns(),
                    block.input.row(row - route_begin));
            }
            block.output.reset(route_end - route_begin, model.descriptor.hidden_size, false);
            blocks.push_back(std::move(block));
            route_begin = route_end;
        }
    }

    if (blocks.empty())
        return Error{ErrorCode::InternalError, "hybrid expert block plan is empty"};
    block_aggregated.assign(blocks.size(), 0);

    const auto add_cpu_work = [&](size_t active_index) {
        if (cpu_ready[active_index] == 0)
        {
            const ActiveExpertExecution& active = layer_state.active_experts[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            bool weights_available = active.lease.gate_up != nullptr;
            if (!weights_available
                && expert.gate_up_weight != invalid_tensor_handle
                && expert.down_weight != invalid_tensor_handle)
            {
                const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                const TensorData& down = model.weights.at(expert.down_weight);
                weights_available = gate_up.dtype == DType::BFloat16
                                    && down.dtype == DType::BFloat16
                                    && gate_up.mapped_data != nullptr
                                    && down.mapped_data != nullptr;
            }
            // 1 means run_hybrid_cpu_blocks may use the current tensors;
            // 2 means a bounded host-cache lease must be acquired first.
            cpu_ready[active_index] = weights_available ? 1 : 2;
            cpu_active_indices.push_back(active_index);
        }
    };

    const auto acquire_cpu_weights = [&](std::vector<size_t> active_indices) -> Result<void> {
        while (!active_indices.empty())
        {
            std::vector<ExpertCachePairRequest> cache_requests;
            std::vector<ExpertCacheLease> leases;
            cache_requests.reserve(active_indices.size());
            leases.resize(active_indices.size());
            for (size_t active_index : active_indices)
            {
                const ExpertPlan& expert = moe.experts[layer_state.active_experts[active_index].batch.expert_id];
                cache_requests.push_back({&model.weights.at(expert.gate_up_weight),
                                          &model.weights.at(expert.down_weight),
                                          residency_group,
                                          expert.cache_key,
                                          victim_metadata(
                                              model,
                                              expert,
                                              layer_state.normalized.rows())});
            }
            auto acquired = model.expert_cache->wait_acquire_ready_pairs(
                cache_requests,
                leases,
                true);
            if (!acquired)
                return acquired.error();
            if (acquired.value() == 0)
                return Error{ErrorCode::InternalError, "hybrid CPU block cache wait made no progress"};

            std::vector<size_t> remaining;
            remaining.reserve(active_indices.size() - acquired.value());
            for (size_t lease_index = 0; lease_index < active_indices.size(); ++lease_index)
            {
                const size_t active_index = active_indices[lease_index];
                if (!leases[lease_index].gate_up)
                {
                    remaining.push_back(active_index);
                    continue;
                }
                ActiveExpertExecution& active = layer_state.active_experts[active_index];
                active.lease = std::move(leases[lease_index]);
                cpu_ready[active_index] = 1;
            }
            active_indices.swap(remaining);
        }
        return {};
    };

    std::vector<size_t> pending_cpu;
    pending_cpu.reserve(active_expert_count);
    cpu_active_indices.reserve(active_expert_count);
    for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
    {
        cpu_active_indices.push_back(active_index);
        ActiveExpertExecution& active = layer_state.active_experts[active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        const TensorData* gate_up = expert.gate_up_weight == invalid_tensor_handle
                                        ? nullptr
                                        : &model.weights.at(expert.gate_up_weight);
        if (!model.expert_cache || !gate_up || expert.cache_key.empty())
        {
            cpu_ready[active_index] = 1;
            continue;
        }
        pending_cpu.push_back(active_index);
    }
    if (!pending_cpu.empty())
    {
        bool release_after_admit = true;
        for (size_t active_index : pending_cpu)
        {
            const ExpertPlan& expert = moe.experts[layer_state.active_experts[active_index].batch.expert_id];
            if (expert.gate_up_weight == invalid_tensor_handle
                || expert.down_weight == invalid_tensor_handle)
            {
                release_after_admit = false;
                break;
            }
            const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
            const TensorData& down = model.weights.at(expert.down_weight);
            const bool mapped_bfloat16 = gate_up.dtype == DType::BFloat16
                                         && down.dtype == DType::BFloat16
                                         && gate_up.mapped_data != nullptr
                                         && down.mapped_data != nullptr;
            const bool file_backed_mxfp4 = gate_up.dtype == DType::MxFp4
                                           && down.dtype == DType::MxFp4
                                           && gate_up.mxfp4_file_storage != nullptr
                                           && down.mxfp4_file_storage != nullptr;
            if (!mapped_bfloat16 && !file_backed_mxfp4)
            {
                release_after_admit = false;
                break;
            }
        }

        if (!release_after_admit)
        {
            auto acquired = acquire_cpu_weights(pending_cpu);
            if (!acquired)
                return acquired.error();
        }
        else
        {
            // File-backed Expert tensors are acquired one at a time solely to
            // hand the bytes to the asynchronous Vulkan admission worker,
            // then released before the full routed wave is submitted. Holding
            // every routed Expert lease would exceed a small host cache when
            // a prefill touches hundreds of the 512 Experts.
            for (size_t active_index : pending_cpu)
            {
                const ExpertPlan& expert = moe.experts[layer_state.active_experts[active_index].batch.expert_id];
                const auto wait_start = std::chrono::steady_clock::now();
                auto lease = model.expert_cache->acquire_pair(
                    model.weights.at(expert.gate_up_weight),
                    model.weights.at(expert.down_weight),
                    residency_group,
                    expert.cache_key,
                    victim_metadata(
                        model,
                        expert,
                        layer_state.normalized.rows()));
                layer_state.active_experts[active_index].metrics.cache_wait_time_microseconds += elapsed_microseconds(wait_start);
                if (!lease)
                    return lease.error();

                ActiveExpertExecution& active = layer_state.active_experts[active_index];
                active.lease = std::move(lease).value();
                const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                const TensorData& down = model.weights.at(expert.down_weight);
                const bool mapped_bfloat16 = gate_up.dtype == DType::BFloat16
                                             && down.dtype == DType::BFloat16
                                             && gate_up.mapped_data != nullptr
                                             && down.mapped_data != nullptr;
                cpu_ready[active_index] = mapped_bfloat16 ? 1 : 0;
                admit_vulkan_expert(
                    model,
                    expert,
                    active.lease,
                    residency_group,
                    static_cast<uint32_t>(layer_state.normalized.rows()),
                    ExecutionBackend::Vulkan);
                active.lease = {};
            }
        }
    }
    for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
    {
        const ActiveExpertExecution& active = layer_state.active_experts[active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        admit_vulkan_expert(
            model,
            expert,
            active.lease,
            residency_group,
            static_cast<uint32_t>(layer_state.normalized.rows()),
            ExecutionBackend::Vulkan);
    }
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    {
        HybridBlockState& block = blocks[block_index];
        const ActiveExpertExecution& active = layer_state.active_experts[block.active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        ExpertBackendRequest request{
            expert.cache_key,
            &block.input,
            &block.output,
            expert.weight_size};
        request.route_aggregation.output = &scratch.backend_aggregated_output;
        request.route_aggregation.routes = std::span<const ExpertRoute>(active.batch.routes).subspan(block.input_begin, block.input.rows());
        request.route_aggregation.token_count = static_cast<uint32_t>(layer_state.normalized.rows());
        request.route_aggregation.completed = &block_aggregated[block_index];
        request.route_aggregation.require_all_requests = true;
        requests.push_back(request);
    }

    const auto backend_execution_start = std::chrono::steady_clock::now();
    std::unique_ptr<ExpertSubmission> submission = model.expert_backend->submit_batch(requests);
    if (!submission)
        return Error{ErrorCode::InternalError, "hybrid expert block submission failed"};

    const std::span<const ExpertBackendExecutionResult> planned = submission->reservations();
    if (planned.size() != requests.size())
    {
        submission->abort();
        return Error{ErrorCode::InternalError, "hybrid expert block reservation shape mismatch"};
    }
    uint32_t backend_max_token_count = 0;
    uint64_t backend_total_weight_bytes = 0;
    uint64_t backend_accelerated_weight_bytes = 0;
    std::vector<uint8_t> active_accelerated(active_expert_count, 0);
    for (size_t request_index = 0; request_index < planned.size(); ++request_index)
    {
        HybridBlockState& block = blocks[request_index];
        const ActiveExpertExecution& active = layer_state.active_experts[block.active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        backend_max_token_count = std::max<uint32_t>(
            backend_max_token_count,
            static_cast<uint32_t>(block.input.rows()));
        if (active_accelerated[block.active_index] == 0)
        {
            backend_total_weight_bytes += expert.weight_size;
            active_accelerated[block.active_index] = 2;
        }
        if (planned[request_index] == ExpertBackendExecutionResult::Executed)
        {
            block.gpu_planned = true;
        }
        else
        {
            add_cpu_work(block.active_index);
        }
    }

    statistics.expert_cache_management_time_microseconds += elapsed_microseconds(cache_management_start);

    std::vector<uint8_t> single_cpu_ready(active_expert_count, 0);
    const auto run_pending_cpu_blocks = [&]() -> Result<uint64_t> {
        uint64_t elapsed = 0;
        for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
        {
            bool has_pending_blocks = false;
            for (const HybridBlockState& block : blocks)
            {
                if (block.active_index == active_index
                    && !block.gpu_planned
                    && !block.cpu_executed)
                {
                    has_pending_blocks = true;
                    break;
                }
            }
            if (!has_pending_blocks)
                continue;

            if (cpu_ready[active_index] != 1)
            {
                auto acquired = acquire_cpu_weights({active_index});
                if (!acquired)
                    return acquired.error();
            }
            std::fill(single_cpu_ready.begin(), single_cpu_ready.end(), uint8_t{0});
            single_cpu_ready[active_index] = 1;
            size_t pending_before = 0;
            for (const HybridBlockState& block : blocks)
            {
                if (block.active_index == active_index
                    && !block.gpu_planned
                    && !block.cpu_executed)
                {
                    ++pending_before;
                }
            }
            elapsed += run_hybrid_cpu_blocks(
                model,
                moe,
                layer_state,
                blocks,
                single_cpu_ready,
                prefetch);
            size_t pending_after = 0;
            for (const HybridBlockState& block : blocks)
            {
                if (block.active_index == active_index
                    && !block.gpu_planned
                    && !block.cpu_executed)
                {
                    ++pending_after;
                }
            }
            if (pending_after == pending_before)
            {
                return Error{
                    ErrorCode::InternalError,
                    "hybrid CPU Expert fallback produced no output"};
            }
            layer_state.active_experts[active_index].lease = {};
            cpu_ready[active_index] = 0;
        }
        return elapsed;
    };

    auto initial_cpu = run_pending_cpu_blocks();
    if (!initial_cpu)
        return initial_cpu.error();
    const uint64_t initial_cpu_time = initial_cpu.value();
    uint64_t compute_wall_time_microseconds = initial_cpu_time;

    const std::vector<ExpertBackendExecutionResult> results = submission->wait();
    bool contract_valid = results.size() == planned.size();
    if (contract_valid)
    {
        for (size_t request_index = 0; request_index < results.size(); ++request_index)
        {
            if (results[request_index] == ExpertBackendExecutionResult::Executed
                && planned[request_index] != ExpertBackendExecutionResult::Executed)
            {
                contract_valid = false;
                break;
            }
        }
    }
    bool has_success = false;
    if (contract_valid)
    {
        for (ExpertBackendExecutionResult result : results)
            has_success = has_success || result == ExpertBackendExecutionResult::Executed;
    }
    bool committed = false;
    if (contract_valid && has_success)
    {
        committed = submission->commit();
        if (!committed)
            submission->abort();
    }
    else
    {
        submission->abort();
    }
    for (size_t request_index = 0; request_index < blocks.size(); ++request_index)
    {
        HybridBlockState& block = blocks[request_index];
        if (!block.gpu_planned)
            continue;
        const ExpertBackendExecutionResult result = request_index < results.size()
                                                        ? results[request_index]
                                                        : ExpertBackendExecutionResult::Failed;
        if (committed && result == ExpertBackendExecutionResult::Executed)
        {
            block.gpu_executed = true;
            block.gpu_aggregated = block_aggregated[request_index] != 0;
            if (active_accelerated[block.active_index] == 2)
            {
                active_accelerated[block.active_index] = 1;
                backend_accelerated_weight_bytes += moe.experts[layer_state.active_experts[block.active_index].batch.expert_id].weight_size;
            }
        }
        else
        {
            block.gpu_planned = false;
            add_cpu_work(block.active_index);
        }
    }

    std::fill(
        scratch.backend_aggregated.begin(),
        scratch.backend_aggregated.end(),
        uint8_t{0});
    std::vector<uint8_t> active_has_blocks(active_expert_count, 0);
    std::vector<uint8_t> active_all_aggregated(active_expert_count, 1);
    bool has_aggregated_output = false;
    for (const HybridBlockState& block : blocks)
    {
        active_has_blocks[block.active_index] = 1;
        has_aggregated_output = has_aggregated_output || block.gpu_aggregated;
        if (!block.gpu_aggregated)
            active_all_aggregated[block.active_index] = 0;
    }
    scratch.backend_aggregated_output_valid = has_aggregated_output;
    for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
    {
        if (active_has_blocks[active_index] != 0
            && active_all_aggregated[active_index] != 0)
        {
            scratch.backend_aggregated[active_index] = 1;
        }
    }

    auto fallback_cpu = run_pending_cpu_blocks();
    if (!fallback_cpu)
        return fallback_cpu.error();
    const uint64_t fallback_cpu_time = fallback_cpu.value();
    compute_wall_time_microseconds = std::max(
        compute_wall_time_microseconds,
        fallback_cpu_time + initial_cpu_time);

    for (const HybridBlockState& block : blocks)
    {
        if (!block.gpu_executed && !block.cpu_executed)
            return Error{ErrorCode::InternalError, "hybrid expert block produced no output"};
    }
    assemble_hybrid_outputs(
        layer_state,
        blocks,
        std::span<const uint8_t>(scratch.backend_aggregated));

    // Let cold GPU admissions overlap the CPU fallback for this layer, then
    // finish them before advancing so the warmed entries are available to
    // later decode passes.
    model.expert_backend->wait_for_background_work();

    for (size_t active_index : cpu_active_indices)
        layer_state.active_experts[active_index].lease = {};

    compute_wall_time_microseconds = std::max(
        compute_wall_time_microseconds,
        elapsed_microseconds(backend_execution_start));
    statistics.expert_compute_time_microseconds += compute_wall_time_microseconds;
    if (model.expert_backend)
    {
        model.expert_backend->observe_phase(
            backend_max_token_count,
            backend_total_weight_bytes,
            backend_accelerated_weight_bytes,
            elapsed_microseconds(backend_execution_start));
    }
    if (active_expert_count > 1)
        statistics.expert_parallel_tasks += active_expert_count;
    for (const ActiveExpertExecution& active : layer_state.active_experts)
    {
        const ExpertExecutionMetrics& metrics = active.metrics;
        statistics.expert_cache_wait_time_microseconds += metrics.cache_wait_time_microseconds;
        statistics.expert_regroup_time_microseconds += metrics.regroup_time_microseconds;
        if (metrics.hinted_bytes > 0)
        {
            ++statistics.expert_prefetches;
            statistics.expert_prefetch_bytes += metrics.hinted_bytes;
        }
        statistics.mxfp4_decode_gemv_rows += metrics.mxfp4_decode_gemv_rows;
        statistics.mxfp4_prefill_gemm_rows += metrics.mxfp4_prefill_gemm_rows;
        statistics.mxfp4_paired_rows += metrics.mxfp4_paired_rows;
        statistics.mxfp4_fused_gate_up_rows += metrics.mxfp4_fused_gate_up_rows;
        statistics.mxfp4_reused_input_rows += metrics.mxfp4_reused_input_rows;
        for (const ExpertRoute& route : active.batch.routes)
        {
            if (route.rank >= maximum_expert_route_ranks)
                continue;
            ++statistics.expert_route_rank_demands[route.rank];
            statistics.expert_route_rank_demand_queue_time_microseconds[route.rank] += metrics.cache_wait_time_microseconds;
        }
        ++statistics.expert_batches;
    }
    layer_state.experts_executed = true;
    return {};
}

Result<void> forward_moe(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    LayerGraphState& layer_state,
    SessionStatistics& statistics,
    CpuExpertExecutionScratch& scratch,
    uint32_t residency_group,
    ExecutionBackend backend,
    bool prefetch)
{
    if (should_use_hybrid_expert_blocks(model, moe, layer_state, backend))
        return run_hybrid_expert_blocks(
            model,
            moe,
            layer_state,
            statistics,
            scratch,
            residency_group,
            prefetch);

    const size_t active_expert_count = layer_state.active_experts.size();
    uint64_t regroup_element_count = 0;
    for (const ActiveExpertExecution& active : layer_state.active_experts)
    {
        regroup_element_count += static_cast<uint64_t>(active.batch.routes.size()) * layer_state.normalized.columns();
    }
    static constexpr uint64_t minimum_parallel_regroup_elements = 256 * 1024;
    bool parallelize_regroup = false;
    int expert_team_size = 1;
#if defined(_OPENMP)
    expert_team_size = std::min(static_cast<int>(active_expert_count), static_cast<int>(cpu_linear_num_threads()));
    parallelize_regroup = expert_team_size > 1 && regroup_element_count >= minimum_parallel_regroup_elements;
#endif
    const int64_t parallel_expert_count = static_cast<int64_t>(active_expert_count);
#pragma omp parallel for schedule(static) num_threads(expert_team_size) if (parallelize_regroup)
    for (int64_t expert_index = 0; expert_index < parallel_expert_count; ++expert_index)
    {
        ActiveExpertExecution& active = layer_state.active_experts[static_cast<size_t>(expert_index)];
        const auto regroup_start = std::chrono::steady_clock::now();
        gather_tokens(layer_state.normalized, active.batch.routes, active.input);
        active.metrics.regroup_time_microseconds += elapsed_microseconds(regroup_start);
    }

    const auto cache_management_start = std::chrono::steady_clock::now();
    uint64_t compute_wall_time_microseconds = 0;
    std::vector<size_t>& uncached = scratch.uncached_indices;
    std::vector<size_t>& pending = scratch.pending_indices;
    uncached.clear();
    pending.clear();
    uncached.reserve(active_expert_count);
    pending.reserve(active_expert_count);
    std::vector<uint8_t>& backend_executed = scratch.backend_executed;
    std::vector<uint8_t>& backend_aggregated = scratch.backend_aggregated;
    std::vector<size_t>& backend_indices = scratch.backend_indices;
    std::vector<ExpertBackendRequest>& backend_requests = scratch.backend_requests;
    backend_executed.assign(active_expert_count, 0);
    backend_aggregated.assign(active_expert_count, 0);
    scratch.backend_aggregated_output_valid = false;
    scratch.backend_aggregated_output.reset(layer_state.normalized.rows(), model.descriptor.hidden_size, true);
    backend_indices.clear();
    backend_requests.clear();
    backend_indices.reserve(active_expert_count);
    backend_requests.reserve(active_expert_count);
    std::unique_ptr<ExpertSubmission> backend_submission;
    std::chrono::steady_clock::time_point backend_execution_start;
    bool backend_reserved_work = false;
    uint32_t backend_max_token_count = 0;
    uint64_t backend_total_weight_bytes = 0;
    uint64_t backend_accelerated_weight_bytes = 0;
    bool backend_reservation_shape_valid = true;
    const bool gpu_expert_batch_eligible = layer_state.normalized.rows() >= vulkan_expert_gpu_min_rows
                                           || model.opt.hybrid_mode == HybridMode::HybridExperts;
    if (backend == ExecutionBackend::Vulkan && model.expert_backend && gpu_expert_batch_eligible)
    {
        for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
        {
            ActiveExpertExecution& active = layer_state.active_experts[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            if (expert.gate_up_weight == invalid_tensor_handle
                || expert.down_weight == invalid_tensor_handle
                || !can_run_vulkan_expert(
                    expert,
                    model.weights.at(expert.gate_up_weight),
                    model.weights.at(expert.down_weight),
                    model.opt.optimization_flags))
            {
                continue;
            }
            const TensorData& gate_up_weight = model.weights.at(expert.gate_up_weight);
            const TensorData& down_weight = model.weights.at(expert.down_weight);
            if (is_qnk_dtype(gate_up_weight.dtype)
                && expert.cache_key.empty())
            {
                continue;
            }
            if (is_qnk_dtype(gate_up_weight.dtype))
            {
                model.expert_backend->admit(
                    expert.cache_key,
                    borrow_resident_tensor(gate_up_weight),
                    expert.gate_up_bias == invalid_tensor_handle
                        ? nullptr
                        : &model.weights.at(expert.gate_up_bias),
                    borrow_resident_tensor(down_weight),
                    expert.down_bias == invalid_tensor_handle
                        ? nullptr
                        : &model.weights.at(expert.down_bias),
                    residency_group,
                    expert.activation_limit,
                    expert.activation);
            }
            backend_indices.push_back(active_index);
            backend_max_token_count = std::max<uint32_t>(backend_max_token_count, static_cast<uint32_t>(active.input.rows()));
            backend_total_weight_bytes += expert.weight_size;
            ExpertBackendRequest request{expert.cache_key, &active.input, &active.output, expert.weight_size};
            request.route_aggregation.output = &scratch.backend_aggregated_output;
            request.route_aggregation.routes = active.batch.routes;
            request.route_aggregation.token_count = static_cast<uint32_t>(layer_state.normalized.rows());
            request.route_aggregation.completed = &backend_aggregated[active_index];
            backend_requests.push_back(request);
        }
        const bool complete_backend_coverage = backend_requests.size() == active_expert_count;
        for (ExpertBackendRequest& request : backend_requests)
        {
            if (complete_backend_coverage)
                request.route_aggregation.require_all_requests = true;
            else
                request.route_aggregation = {};
        }
        backend_execution_start = std::chrono::steady_clock::now();
        backend_submission = model.expert_backend->submit_batch(backend_requests);
        if (backend_submission)
        {
            const std::span<const ExpertBackendExecutionResult> planned = backend_submission->reservations();
            backend_reservation_shape_valid = planned.size() == backend_indices.size();
            if (!backend_reservation_shape_valid)
            {
                // A shape mismatch invalidates the batch and enables CPU retry.
                backend_submission->abort();
            }
            else
            {
                for (size_t result_index = 0; result_index < planned.size(); ++result_index)
                {
                    if (planned[result_index] != ExpertBackendExecutionResult ::Executed)
                        continue;
                    backend_executed[backend_indices[result_index]] = 1;
                    const ActiveExpertExecution& active = layer_state.active_experts[backend_indices[result_index]];
                    backend_accelerated_weight_bytes += moe.experts[active.batch.expert_id].weight_size;
                    backend_reserved_work = true;
                }
            }
        }
    }

    for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
    {
        ActiveExpertExecution& active = layer_state.active_experts[active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        if (backend_executed[active_index])
            continue;
        const TensorData* gate_up = expert.gate_up_weight == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_weight);
        if (!model.expert_cache || !gate_up || expert.cache_key.empty())
        {
            uncached.push_back(active_index);
            continue;
        }
        pending.push_back(active_index);
    }

    bool ready_batch_acquired = false;
    if (!pending.empty())
    {
        std::vector<ExpertCachePairRequest>& requests = scratch.cache_requests;
        std::vector<ExpertCacheLease>& leases = scratch.cache_leases;
        requests.clear();
        leases.clear();
        leases.resize(pending.size());
        requests.reserve(pending.size());
        for (size_t active_index : pending)
        {
            const ExpertPlan& expert = moe.experts[layer_state.active_experts[active_index].batch.expert_id];
            requests.push_back({
                &model.weights.at(expert.gate_up_weight),
                &model.weights.at(expert.down_weight),
                residency_group,
                expert.cache_key,
                victim_metadata(model, expert, layer_state.normalized.rows()),
            });
        }
        auto ready = model.expert_cache->try_acquire_ready_pairs(requests, leases);
        if (!ready)
            return ready.error();
        ready_batch_acquired = ready.value();
        if (ready_batch_acquired)
        {
            for (size_t pending_index = 0; pending_index < pending.size(); ++pending_index)
            {
                ActiveExpertExecution& active = layer_state.active_experts[pending[pending_index]];
                active.lease = std::move(leases[pending_index]);
                const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                admit_vulkan_expert(
                    model,
                    expert,
                    active.lease,
                    residency_group,
                    static_cast<uint32_t>(layer_state.normalized.rows()),
                    backend);
            }
        }
    }

    statistics.expert_cache_management_time_microseconds += elapsed_microseconds(cache_management_start);
    compute_wall_time_microseconds += run_experts(model, moe, layer_state, uncached, scratch, prefetch);

    if (ready_batch_acquired)
    {
        compute_wall_time_microseconds += run_experts(model, moe, layer_state, pending, scratch, prefetch);
        const auto lease_release_start = std::chrono::steady_clock::now();
        for (size_t active_index : pending)
            layer_state.active_experts[active_index].lease = {};
        pending.clear();
        statistics.expert_cache_management_time_microseconds += elapsed_microseconds(lease_release_start);
    }

    // The wait path admits only the subset that currently fits. This keeps
    // large prefill batches from pinning more file-backed Expert pairs than
    // the cache can hold at once.
    while (!pending.empty())
    {
        std::vector<size_t>& ready_indices = scratch.ready_indices;
        ready_indices.clear();
        ready_indices.reserve(pending.size());

        std::vector<ExpertCachePairRequest>& requests = scratch.cache_requests;
        std::vector<ExpertCacheLease>& leases = scratch.cache_leases;
        requests.clear();
        leases.clear();
        requests.reserve(pending.size());
        leases.resize(pending.size());
        for (size_t active_index : pending)
        {
            const ExpertPlan& expert = moe.experts[layer_state.active_experts[active_index].batch.expert_id];
            requests.push_back({
                &model.weights.at(expert.gate_up_weight),
                &model.weights.at(expert.down_weight),
                residency_group,
                expert.cache_key,
                victim_metadata(model, expert, layer_state.normalized.rows()),
            });
        }

        const auto cache_wait_start = std::chrono::steady_clock::now();
        auto acquired = model.expert_cache->wait_acquire_ready_pairs(requests, leases, true);
        const uint64_t cache_wait_microseconds = elapsed_microseconds(cache_wait_start);
        if (!acquired)
            return acquired.error();
        if (acquired.value() == 0)
        {
            return Error{ErrorCode::InternalError, "Expert cache ready wait acquired no pairs"};
        }

        bool wait_accounted = false;
        size_t pending_count = 0;
        for (size_t pending_index = 0; pending_index < pending.size(); ++pending_index)
        {
            ExpertCacheLease& lease = leases[pending_index];
            if (!lease.gate_up)
            {
                pending[pending_count++] = pending[pending_index];
                continue;
            }
            const size_t active_index = pending[pending_index];
            ActiveExpertExecution& active = layer_state.active_experts[active_index];
            active.lease = std::move(lease);
            if (!wait_accounted)
            {
                active.metrics.cache_wait_time_microseconds += cache_wait_microseconds;
                wait_accounted = true;
            }
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            admit_vulkan_expert(
                model,
                expert,
                active.lease,
                residency_group,
                static_cast<uint32_t>(layer_state.normalized.rows()),
                backend);
            ready_indices.push_back(active_index);
        }
        pending.resize(pending_count);
        compute_wall_time_microseconds += run_experts(model, moe, layer_state, ready_indices, scratch, prefetch);
        for (size_t active_index : ready_indices)
            layer_state.active_experts[active_index].lease = {};
    }

    if (backend_submission)
    {
        const std::vector<ExpertBackendExecutionResult> backend_results = backend_submission->wait();
        bool backend_result_contract_valid = backend_reservation_shape_valid && backend_results.size() == backend_indices.size();
        if (backend_result_contract_valid)
        {
            const std::span<const ExpertBackendExecutionResult> planned = backend_submission->reservations();
            for (size_t result_index = 0; result_index < backend_results.size(); ++result_index)
            {
                if (backend_results[result_index] == ExpertBackendExecutionResult::Executed
                    && planned[result_index] != ExpertBackendExecutionResult::Executed)
                {
                    backend_result_contract_valid = false;
                    break;
                }
            }
        }
        bool backend_has_executed = false;
        if (backend_result_contract_valid)
        {
            for (ExpertBackendExecutionResult result : backend_results)
                backend_has_executed = backend_has_executed || result == ExpertBackendExecutionResult::Executed;
        }
        bool backend_commit_succeeded = backend_result_contract_valid && !backend_has_executed;
        if (backend_has_executed)
        {
            backend_commit_succeeded = backend_submission->commit();
            if (!backend_commit_succeeded)
                backend_submission->abort();
        }
        else
        {
            backend_submission->abort();
        }
        if (!backend_commit_succeeded)
            std::fill(backend_aggregated.begin(), backend_aggregated.end(), uint8_t{0});
        scratch.backend_aggregated_output_valid = std::any_of(
            backend_aggregated.begin(),
            backend_aggregated.end(),
            [](uint8_t value) { return value != 0; });
        if (backend_reserved_work)
        {
            compute_wall_time_microseconds = std::max(compute_wall_time_microseconds, elapsed_microseconds(backend_execution_start));
        }
        std::vector<size_t>& failed_indices = scratch.failed_indices;
        failed_indices.clear();
        failed_indices.reserve(backend_indices.size());
        for (size_t result_index = 0; result_index < backend_indices.size(); ++result_index)
        {
            const size_t active_index = backend_indices[result_index];
            if (!backend_executed[active_index])
                continue;
            const ExpertBackendExecutionResult backend_result = result_index < backend_results.size()
                                                                    ? backend_results[result_index]
                                                                    : ExpertBackendExecutionResult ::Failed;
            if (!backend_commit_succeeded
                || backend_result != ExpertBackendExecutionResult ::Executed)
            {
                backend_executed[active_index] = 0;
                const ActiveExpertExecution& active = layer_state.active_experts[active_index];
                backend_accelerated_weight_bytes -= std::min<uint64_t>(backend_accelerated_weight_bytes, moe.experts[active.batch.expert_id].weight_size);
                failed_indices.push_back(active_index);
                continue;
            }

            ActiveExpertExecution& active = layer_state.active_experts[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
            const TensorData& down = model.weights.at(expert.down_weight);
            record_mxfp4(gate_up, active.input.rows(), active.metrics);
            record_mxfp4(down, active.input.rows(), active.metrics);
            active.metrics.mxfp4_fused_gate_up_rows += static_cast<uint64_t>(active.input.rows()) * gate_up.shape[0] / 2;
        }

        if (!failed_indices.empty())
        {
            const auto fallback_cache_start = std::chrono::steady_clock::now();
            for (size_t active_index : failed_indices)
            {
                ActiveExpertExecution& active = layer_state.active_experts[active_index];
                const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                const TensorData& down = model.weights.at(expert.down_weight);
                if (model.expert_cache && !expert.cache_key.empty())
                {
                    const auto wait_start = std::chrono::steady_clock::now();
                    auto lease = model.expert_cache->acquire_pair(
                        gate_up,
                        down,
                        residency_group,
                        expert.cache_key,
                        victim_metadata(model, expert, layer_state.normalized.rows()));
                    active.metrics.cache_wait_time_microseconds += elapsed_microseconds(wait_start);
                    if (!lease)
                    {
                        return lease.error();
                    }
                    active.lease = std::move(lease).value();
                    admit_vulkan_expert(
                        model,
                        expert,
                        active.lease,
                        residency_group,
                        static_cast<uint32_t>(layer_state.normalized.rows()),
                        backend);
                }
            }
            statistics.expert_cache_management_time_microseconds += elapsed_microseconds(fallback_cache_start);
            compute_wall_time_microseconds += run_experts(model, moe, layer_state, failed_indices, scratch, prefetch);
            for (size_t active_index : failed_indices)
            {
                layer_state.active_experts[active_index].lease = {};
            }
        }
        model.expert_backend->observe_phase(backend_max_token_count, backend_total_weight_bytes, backend_accelerated_weight_bytes, elapsed_microseconds(backend_execution_start));
    }

    statistics.expert_compute_time_microseconds += compute_wall_time_microseconds;
    if (expert_team_size > 1)
        statistics.expert_parallel_tasks += active_expert_count;
    for (const ActiveExpertExecution& active : layer_state.active_experts)
    {
        const ExpertExecutionMetrics& metrics = active.metrics;
        statistics.expert_cache_wait_time_microseconds += metrics.cache_wait_time_microseconds;
        statistics.expert_regroup_time_microseconds += metrics.regroup_time_microseconds;
        if (metrics.hinted_bytes > 0)
        {
            ++statistics.expert_prefetches;
            statistics.expert_prefetch_bytes += metrics.hinted_bytes;
        }
        statistics.mxfp4_decode_gemv_rows += metrics.mxfp4_decode_gemv_rows;
        statistics.mxfp4_prefill_gemm_rows += metrics.mxfp4_prefill_gemm_rows;
        statistics.mxfp4_paired_rows += metrics.mxfp4_paired_rows;
        statistics.mxfp4_fused_gate_up_rows += metrics.mxfp4_fused_gate_up_rows;
        statistics.mxfp4_reused_input_rows += metrics.mxfp4_reused_input_rows;
        for (const ExpertRoute& route : active.batch.routes)
        {
            if (route.rank >= maximum_expert_route_ranks)
                continue;
            ++statistics.expert_route_rank_demands[route.rank];
            statistics.expert_route_rank_demand_queue_time_microseconds[route.rank] += metrics.cache_wait_time_microseconds;
        }
        ++statistics.expert_batches;
    }
    layer_state.experts_executed = true;
    return {};
}

bool initialize_backend_aggregated_output(
    const CpuExpertExecutionScratch& scratch,
    size_t rows,
    uint32_t columns,
    CpuBatch& output)
{
    if (!scratch.backend_aggregated_output_valid
        || scratch.backend_aggregated_output.rows() != rows
        || scratch.backend_aggregated_output.columns() != columns)
    {
        output.reset(rows, columns, true);
        return false;
    }

    output.reset(rows, columns, false);
    for (size_t row = 0; row < rows; ++row)
    {
        std::copy_n(scratch.backend_aggregated_output.row(row), columns, output.row(row));
    }
    return true;
}

} // namespace moe
} // namespace ncnn
