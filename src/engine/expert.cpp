#include "expert.h"

#include "sessionstate.h"
#include "expertbackend.h"
#include "graph/compiledmodel.h"
#include "kernels/bfloat16.h"
#include "kernels/fastmath.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"
#include "kernels/vector.h"
#include "storage/expertcache.h"
#include "storage/weightstore.h"
#include "backends/ncnn/linear.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
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

static void prefetch_buffer(const void* data, size_t size)
{
    if (!data || size == 0)
        return;
    const size_t hinted_size = std::min(size, expert_prefetch_limit_size);
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    for (size_t offset = 0; offset < hinted_size; offset += cache_line_size)
        prefetch_address(ptr + offset);
}

static void prefetch_tensor(const TensorData& tensor)
{
    if (tensor.dtype == DType::Float32)
    {
        const std::span<const float> values = tensor.float32_values();
        prefetch_buffer(values.data(), values.size() * sizeof(float));
        return;
    }
    if (tensor.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> values = tensor.bfloat16_values();
        prefetch_buffer(values.data(), values.size() * sizeof(uint16_t));
        return;
    }
    if (tensor.dtype == DType::Int8)
    {
        const std::span<const int8_t> values = tensor.int8_values();
        prefetch_buffer(values.data(), values.size());
        return;
    }
    if (tensor.dtype == DType::Float8E4M3)
    {
        const std::span<const uint8_t> values = tensor.float8_values();
        prefetch_buffer(values.data(), values.size());
        return;
    }
    if (tensor.dtype == DType::Int64)
    {
        const std::span<const int64_t> values = tensor.int64_values();
        prefetch_buffer(values.data(), values.size() * sizeof(int64_t));
        return;
    }
    if (tensor.dtype == DType::MxFp4)
    {
        prefetch_buffer(tensor.mxfp4_blocks.data(), tensor.mxfp4_blocks.size());
        prefetch_buffer(tensor.mxfp4_scales.data(), tensor.mxfp4_scales.size());
        return;
    }
    if (is_qnk_dtype(tensor.dtype))
    {
        const std::span<const uint8_t> values = tensor.qnk_values();
        prefetch_buffer(values.data(), values.size());
        return;
    }
    return;
}

static void prefetch_weight(const WeightStore& weights, TensorHandle handle)
{
    if (handle != invalid_tensor_handle) prefetch_tensor(weights.at(handle));
}

static float activate(float value,
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

static void expert_linear(const TensorData& matrix,
                          const TensorData* bias,
                          const CompiledOperator* executable,
                          const ActivationBuffer& input,
                          ActivationBuffer& output,
                          uint64_t optimization_flags,
                          ActivationBuffer* quantized_input_scratch)
{
    if (bias)
        forward_linear(matrix, *bias, input, output, optimization_flags, executable,
                       ExecutionBackend::Cpu, quantized_input_scratch);
    else
        forward_linear(matrix, input, output, optimization_flags, executable,
                       ExecutionBackend::Cpu, quantized_input_scratch);
}

static void forward_expert(const CompiledModel& model,
                           const ExpertPlan& expert,
                           const ExpertCacheLease* cached_weights,
                           const ActivationBuffer& input,
                           ActivationBuffer& output,
                           ExpertWorkspace& workspace,
                           bool prefetch)
{
    assert(&input != &output);
    const WeightStore& weights = model.weights;
    const CompiledOperatorTable& operators = model.operators;
    const uint64_t optimization_flags = model.opt.optimization_flags;
    if (expert.gate_up_weight != invalid_tensor_handle)
    {
        const TensorData& gate_up_weight = cached_weights && cached_weights->gate_up ? *cached_weights->gate_up : weights.at(expert.gate_up_weight);
        const CompiledOperator* gate_up_operator = cached_weights && cached_weights->gate_up
                                                       ? cached_weights->gate_up_operator
                                                       : operators.find_weight(expert.gate_up_weight);
        if (prefetch)
            prefetch_tensor(gate_up_weight);
        const TensorData* gate_up_bias = expert.gate_up_bias == invalid_tensor_handle ? nullptr : &weights.at(expert.gate_up_bias);
        ActivationBuffer& activated = workspace.projection;
        if (gate_up_weight.dtype == DType::MxFp4)
        {
            forward_gate_up_mxfp4(gate_up_weight,
                                  gate_up_bias,
                                  input,
                                  expert.activation,
                                  expert.activation_limit,
                                  activated,
                                  optimization_flags);
        }
        else if (expert.layout == ExpertLayout::PackedGateUpDown)
        {
            expert_linear(gate_up_weight, gate_up_bias, gate_up_operator, input, activated,
                          optimization_flags, &workspace.quantized_input);
            const size_t rows = activated.rows();
            const uint32_t intermediate_size = activated.columns() / 2;
            const bool vector_silu = intermediate_size >= 4
                                     && expert.activation_limit <= 0.0f
                                     && has_flag(optimization_flags, OptimizationCpuFastSilu)
                                     && (expert.activation == ExpertActivation::Silu
                                         || expert.activation == ExpertActivation::DeepSeekSwiGlu);
            // Compact forward while source rows still use the full gate/up stride.
            for (size_t token_index = 0; token_index < rows; ++token_index)
            {
                const float* source = activated.row(token_index);
                float* destination = activated.row(0) + token_index * intermediate_size;
                if (vector_silu)
                {
                    // FastSilu's stable negative-gate form and this vector
                    // form are algebraically equal but round differently.
                    float_silu_mul(destination,
                                   source,
                                   source + intermediate_size,
                                   1.0f,
                                   0.0f,
                                   intermediate_size);
                    continue;
                }
                for (uint32_t column = 0; column < intermediate_size; ++column)
                {
                    const float gate = source[column];
                    const float up = source[intermediate_size + column];
                    destination[column] = activate(gate,
                                                   expert.activation,
                                                   expert.activation_limit,
                                                   optimization_flags)
                                          * up;
                }
            }
            activated.reset(rows, intermediate_size, false);
        }
        else
        {
            expert_linear(gate_up_weight, gate_up_bias, gate_up_operator, input, activated,
                          optimization_flags, &workspace.quantized_input);
            const size_t rows = activated.rows();
            const uint32_t intermediate_size = activated.columns() / 2;
            for (size_t token_index = 0; token_index < rows; ++token_index)
            {
                const float* source = activated.row(token_index);
                float* destination = activated.row(0) + token_index * intermediate_size;
                for (uint32_t column = 0; column < intermediate_size; ++column)
                {
                    const float gate = expert.activation_limit > 0.0f ? std::min(source[column * 2], expert.activation_limit) : source[column * 2];
                    const float linear = expert.activation_limit > 0.0f
                                             ? std::clamp(source[column * 2 + 1], -expert.activation_limit, expert.activation_limit)
                                             : source[column * 2 + 1];
                    const float silu = scaled_silu(gate,
                                                   1.702f,
                                                   optimization_flags);
                    destination[column] = silu * (linear + 1.0f);
                }
            }
            activated.reset(rows, intermediate_size, false);
        }
        const TensorData& down_weight = cached_weights && cached_weights->down ? *cached_weights->down : weights.at(expert.down_weight);
        const CompiledOperator* down_operator = cached_weights && cached_weights->down
                                                    ? cached_weights->down_operator
                                                    : operators.find_weight(expert.down_weight);
        if (prefetch)
            prefetch_tensor(down_weight);
        expert_linear(down_weight,
                      expert.down_bias == invalid_tensor_handle ? nullptr : &weights.at(expert.down_bias),
                      down_operator,
                      activated,
                      output,
                      optimization_flags,
                      &workspace.quantized_input);
        return;
    }

    if (prefetch)
        prefetch_weight(weights, expert.up_weight);
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
                prefetch_weight(weights, expert.gate_weight);
                gate_prefetched = true;
            }
            ActivationBuffer& activated = workspace.projection;
            if (forward_gate_up_float8(gate_weight,
                                       up_weight,
                                       input,
                                       expert.activation,
                                       expert.activation_limit,
                                       activated,
                                       optimization_flags,
                                       gate_operator,
                                       up_operator,
                                       &workspace.quantized_input))
            {
                if (prefetch)
                {
                    prefetch_weight(weights, expert.down_weight);
                }
                expert_linear(down_weight,
                              nullptr,
                              down_operator,
                              activated,
                              output,
                              optimization_flags,
                              &workspace.quantized_input);
                return;
            }
        }
    }
    ActivationBuffer& up = workspace.projection;
    expert_linear(up_weight,
                  nullptr,
                  operators.find_weight(expert.up_weight),
                  input,
                  up,
                  optimization_flags,
                  &workspace.quantized_input);
    if (expert.layout != ExpertLayout::UpDown)
    {
        if (prefetch && !gate_prefetched)
            prefetch_weight(weights, expert.gate_weight);
        ActivationBuffer& gate = workspace.gate;
        expert_linear(weights.at(expert.gate_weight),
                      nullptr,
                      operators.find_weight(expert.gate_weight),
                      input,
                      gate,
                      optimization_flags,
                      &workspace.quantized_input);
        const bool vector_silu = up.columns() >= 4
                                 && expert.activation_limit <= 0.0f
                                 && has_flag(optimization_flags, OptimizationCpuFastSilu)
                                 && (expert.activation == ExpertActivation::Silu
                                     || expert.activation == ExpertActivation::DeepSeekSwiGlu);
        for (size_t token_index = 0; token_index < up.rows(); ++token_index)
        {
            float* up_row = up.row(token_index);
            const float* gate_row = gate.row(token_index);
            if (vector_silu)
            {
                float_silu_mul(up_row,
                               gate_row,
                               up_row,
                               1.0f,
                               0.0f,
                               up.columns());
                continue;
            }
            for (uint32_t column = 0; column < up.columns(); ++column)
            {
                if (expert.activation == ExpertActivation::DeepSeekSwiGlu && expert.activation_limit > 0.0f)
                    up_row[column] = std::clamp(up_row[column], -expert.activation_limit, expert.activation_limit);
                up_row[column] *= activate(gate_row[column],
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
                token[column] = activate(token[column],
                                         expert.activation,
                                         expert.activation_limit,
                                         optimization_flags);
        }
    }
    if (prefetch)
        prefetch_weight(weights, expert.down_weight);
    expert_linear(weights.at(expert.down_weight),
                  nullptr,
                  operators.find_weight(expert.down_weight),
                  up,
                  output,
                  optimization_flags,
                  &workspace.quantized_input);
}

void forward_shared_expert(const CompiledModel& model,
                           const MoeBlockPlan& moe,
                           const ActivationBuffer& input,
                           ActivationBuffer& output,
                           ExpertWorkspace& workspace,
                           const std::shared_ptr<const DeviceTensor_vulkan>& device_input,
                           std::shared_ptr<const DeviceTensor_vulkan>* device_output)
{
    assert(&input != &output);
    if (device_output)
        device_output->reset();
    const ExpertPlan& expert = moe.shared_expert;
    const uint64_t optimization_flags = model.opt.optimization_flags;
    const bool has_router_gate = moe.shared_expert_gate_weight != invalid_tensor_handle;
    const auto& float8_gate = model.operators.at_weight(expert.gate_weight).float8;
    const auto& float8_up = model.operators.at_weight(expert.up_weight).float8;
    const auto& float8_down = model.operators.at_weight(expert.down_weight).float8;
    if (!has_router_gate && float8_gate && float8_up && float8_down && expert.activation == ExpertActivation::DeepSeekSwiGlu)
    {
        if (device_output && device_input && !device_input->empty())
        {
            auto completed = std::make_shared<DeviceTensor_vulkan>();
            if (float8_gate->forward_swiglu_chain_device(*device_input, *float8_up, *float8_down, expert.activation, expert.activation_limit, *completed))
            {
                output.clear();
                *device_output = std::move(completed);
                return;
            }
        }
        if (float8_gate->forward_swiglu_chain(input, *float8_up, *float8_down, expert.activation, expert.activation_limit, output))
            return;
    }
    const CompiledOperator& fused_shared_operator = model.operators.at(moe.fused_shared_input_bfloat16_operator);
    if (fused_shared_operator.bfloat16
        && expert.down_weight != invalid_tensor_handle)
    {
        const CompiledOperator& down_operator = model.operators.at_weight(expert.down_weight);
        if (down_operator.bfloat16)
        {
            const uint32_t intermediate = model.weights.at(expert.up_weight).shape[0];
            if (fused_shared_operator.bfloat16->forward_swiglu_chain(input,
                                                                     *down_operator.bfloat16,
                                                                     intermediate,
                                                                     expert.activation,
                                                                     expert.activation_limit,
                                                                     has_router_gate,
                                                                     output))
            {
                return;
            }
        }
    }
    if (fused_shared_operator.bfloat16)
    {
        ActivationBuffer& fused = workspace.projection;
        if (fused_shared_operator.bfloat16->forward(input,
                                                    fused))
        {
            const uint32_t intermediate = model.weights.at(expert.up_weight).shape[0];
            const uint32_t expected_columns = intermediate * 2 + (has_router_gate ? 1 : 0);
            if (fused.columns() == expected_columns)
            {
                ActivationBuffer& activated = workspace.gate;
                activated.reset(fused.rows(), intermediate, false);
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
                        destination[column] = activate(source[column],
                                                       expert.activation,
                                                       expert.activation_limit,
                                                       optimization_flags)
                                              * source[intermediate + column];
                    }
                }
                expert_linear(model.weights.at(expert.down_weight),
                              nullptr,
                              model.operators.find_weight(expert.down_weight),
                              activated,
                              output,
                              optimization_flags,
                              &workspace.quantized_input);
                if (has_router_gate)
                {
                    for (size_t token_index = 0;
                         token_index < output.rows();
                         ++token_index)
                    {
                        const float scale = 1.0f
                                            / (1.0f
                                               + float_approximate_exp(-fused.row(token_index)
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
                return;
            }
        }
    }

    forward_expert(model,
                   moe.shared_expert,
                   nullptr,
                   input,
                   output,
                   workspace,
                   false);
    if (moe.shared_expert_gate_weight == invalid_tensor_handle)
        return;
    ActivationBuffer& gate = workspace.gate;
    forward_linear(model.weights.at(moe.shared_expert_gate_weight),
                   input,
                   gate,
                   optimization_flags,
                   model.operators.find_weight(moe.shared_expert_gate_weight),
                   ExecutionBackend::Cpu,
                   &workspace.quantized_input);
    assert(gate.columns() == 1);
    for (size_t token_index = 0; token_index < output.rows(); ++token_index)
    {
        const float scale = 1.0f / (1.0f + float_approximate_exp(-gate.row(token_index)[0]));
        float* token = output.row(token_index);
        for (uint32_t column = 0; column < output.columns(); ++column)
            token[column] *= scale;
    }
}

static bool expert_cache_wait_timing_enabled(const CompiledModel& model) noexcept
{
    // Queue latency drives rank-adaptive router prefetch.
    return has_flag(model.opt.flags, OptionRouterPrediction)
           && has_flag(model.opt.flags, OptionRankAdaptivePrefetch);
}

static std::chrono::steady_clock::time_point expert_cache_wait_start(const CompiledModel& model)
{
    return expert_cache_wait_timing_enabled(model) ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
}

static uint64_t expert_cache_wait_elapsed(std::chrono::steady_clock::time_point start)
{
    if (start == std::chrono::steady_clock::time_point{}) return 0;
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

static void gather_tokens(const ActivationBuffer& source, std::span<const ExpertRoute> routes, ActivationBuffer& gathered)
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
    ActivationBuffer input;
    ActivationBuffer output;
    bool gpu_planned = false;
    bool gpu_executed = false;
    bool gpu_aggregated = false;
    bool cpu_executed = false;
};

static size_t hybrid_block_end(std::span<const ExpertRoute> routes,
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

static void run_hybrid_cpu_blocks(const CompiledModel& model,
                                  const MoeBlockPlan& moe,
                                  LayerState& layer_state,
                                  ExpertWorkspace& workspace,
                                  std::span<HybridBlockState> blocks,
                                  size_t active_index,
                                  bool prefetch)
{
    size_t first_block = blocks.size();
    size_t block_count = 0;
    size_t total_rows = 0;
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    {
        const HybridBlockState& block = blocks[block_index];
        if (!block.gpu_planned && !block.cpu_executed)
        {
            if (block_count == 0)
                first_block = block_index;
            ++block_count;
            total_rows += block.input.rows();
        }
    }
    if (block_count == 0)
        return;

    ExpertState& active = layer_state.active_experts()[active_index];
    const ExpertPlan& expert = moe.experts[active.batch.expert_id];
    // CPU cache leases are acquired one Expert at a time. Keep its blocks
    // batched, and reuse the Expert input only when a merge is needed.
    if (block_count > 1)
    {
        active.input.reset(total_rows, layer_state.normalized.columns(), false);
        size_t row_offset = 0;
        for (const HybridBlockState& block : blocks)
        {
            if (block.gpu_planned || block.cpu_executed)
                continue;
            std::copy(block.input.values().begin(), block.input.values().end(), active.input.row(row_offset));
            row_offset += block.input.rows();
        }
    }

    if (block_count == 1)
    {
        HybridBlockState& block = blocks[first_block];
        forward_expert(model,
                       expert,
                       active.lease.gate_up ? &active.lease : nullptr,
                       block.input,
                       block.output,
                       workspace,
                       prefetch);
        block.cpu_executed = block.output.rows() == block.input.rows()
                             && block.output.columns() != 0;
        return;
    }

    ActivationBuffer output;
    forward_expert(model,
                   expert,
                   active.lease.gate_up ? &active.lease : nullptr,
                   active.input,
                   output,
                   workspace,
                   prefetch);
    if (output.rows() != total_rows || output.columns() == 0)
        return;
    size_t row_offset = 0;
    for (HybridBlockState& block : blocks)
    {
        if (block.gpu_planned || block.cpu_executed)
            continue;
        block.output.reset(block.input.rows(), output.columns(), false);
        std::copy_n(output.row(row_offset), block.output.values().size(), block.output.row(0));
        block.cpu_executed = true;
        row_offset += block.input.rows();
    }

    return;
}

static void assemble_hybrid_outputs(LayerState& layer_state,
                                    std::span<const HybridBlockState> blocks,
                                    std::span<const uint8_t> backend_aggregated)
{
    std::vector<uint8_t> initialized(layer_state.active_experts().size(), 0);
    for (const HybridBlockState& block : blocks)
    {
        if (block.active_index >= layer_state.active_experts().size()
            || block.active_index >= backend_aggregated.size()
            || block.gpu_aggregated
            || (!block.cpu_executed && !block.gpu_executed))
        {
            continue;
        }

        ExpertState& active = layer_state.active_experts()[block.active_index];
        if (initialized[block.active_index] == 0)
        {
            active.output.reset(active.batch.routes.size(),
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
            std::copy_n(block.output.row(row),
                        block.output.columns(),
                        active.output.row(block.input_begin + row));
        }
    }
}

static void forward_experts(const CompiledModel& model,
                            const MoeBlockPlan& moe,
                            LayerState& layer_state,
                            std::span<const size_t> active_indices,
                            ExpertScratch& scratch,
                            bool prefetch)
{
    if (active_indices.empty())
        return;

    // Device-capable routes do not own gathered Host storage until a CPU
    // consumer is actually selected (miss, rejection, or failed GPU commit).
    for (size_t active_index : active_indices)
    {
        ExpertState& active = layer_state.active_experts()[active_index];
        if (active.input.rows() == active.batch.routes.size()
            && active.input.columns() == layer_state.normalized.columns()) continue;
        gather_tokens(layer_state.normalized, active.batch.routes, active.input);
    }
    std::vector<Mxfp4Task>& decode_tasks = scratch.decode_tasks;
    decode_tasks.clear();
    bool prefetched_batch_weights = false;
    const ExpertState& first_active = layer_state.active_experts()[active_indices.front()];
    const ExpertPlan& first_expert = moe.experts[first_active.batch.expert_id];
    const TensorData* first_gate_up = first_active.lease.gate_up.get();
    if (!first_gate_up && first_expert.gate_up_weight != invalid_tensor_handle)
        first_gate_up = &model.weights.at(first_expert.gate_up_weight);
    if (first_gate_up && first_gate_up->dtype == DType::MxFp4)
    {
        decode_tasks.reserve(active_indices.size());
        for (size_t active_index : active_indices)
        {
            ExpertState& active = layer_state.active_experts()[active_index];
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

        prefetched_batch_weights = prefetch
                                   && std::all_of(decode_tasks.begin(),
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
                prefetch_tensor(*decode_tasks[task_index].gate_up);
                prefetch_tensor(*decode_tasks[task_index].down);
            }
        }

        const bool grouped_decode = forward_experts_mxfp4(decode_tasks,
                                                          &scratch.kernels,
                                                          model.opt.optimization_flags);
        if (grouped_decode) return;
    }

    bool parallelize_experts = false;
    int expert_team_size = 1;
#if defined(_OPENMP)
    expert_team_size = std::min(static_cast<int>(active_indices.size()), static_cast<int>(cpu_linear_num_threads()));
    parallelize_experts = expert_team_size > 1;
    if (parallelize_experts
        && expert_team_size <= static_cast<int>(cpu_linear_num_threads() / 2))
    {
        // BF16 Experts may underfill the outer team. Prefer matrix teams
        // only when every projection can use at least twice as many threads.
        bool parallelize_matrices = true;
        for (size_t active_index : active_indices)
        {
            const ExpertState& active = layer_state.active_experts()[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            const TensorData* gate_up = active.lease.gate_up.get();
            if (!gate_up && expert.gate_up_weight != invalid_tensor_handle)
                gate_up = &model.weights.at(expert.gate_up_weight);
            const TensorData* down = active.lease.down.get();
            if (!down && expert.down_weight != invalid_tensor_handle)
                down = &model.weights.at(expert.down_weight);
            for (const TensorData* matrix : {gate_up, down})
            {
                if (!matrix || matrix->shape.size() != 2 || matrix->dtype != DType::BFloat16)
                {
                    parallelize_matrices = false;
                    break;
                }
                const uint64_t operation_count = static_cast<uint64_t>(matrix->shape[0])
                                                 * matrix->shape[1] * active.input.rows();
                if (cpu_linear_team_size(operation_count, matrix->dtype) / 2 < expert_team_size)
                {
                    parallelize_matrices = false;
                    break;
                }
            }
            if (!parallelize_matrices)
                break;
        }
        parallelize_experts = !parallelize_matrices;
    }
#endif
    const int64_t parallel_expert_count = static_cast<int64_t>(active_indices.size());
    Bfloat16BatchedLinearExecutionCounter* const bfloat16_counter = current_bfloat16_batched_linear_execution_counter();
    const size_t workspace_count = parallelize_experts
                                       ? static_cast<size_t>(expert_team_size)
                                       : 1;
    if (scratch.expert_workspaces.size() < workspace_count)
        scratch.expert_workspaces.resize(workspace_count);
#pragma omp parallel for schedule(dynamic, 1) num_threads(expert_team_size) if (parallelize_experts)
    for (int64_t task_index = 0; task_index < parallel_expert_count; ++task_index)
    {
        const ScopedBfloat16BatchedLinearExecutionCounter bfloat16_scope(bfloat16_counter);
        ExpertState& active = layer_state.active_experts()[active_indices[static_cast<size_t>(task_index)]];
        const uint32_t expert_id = active.batch.expert_id;
        size_t workspace_index = 0;
#if defined(_OPENMP)
        if (parallelize_experts)
            workspace_index = static_cast<size_t>(omp_get_thread_num());
#endif
        forward_expert(model,
                       moe.experts[expert_id],
                       active.lease.gate_up ? &active.lease : nullptr,
                       active.input,
                       active.output,
                       scratch.expert_workspaces[workspace_index],
                       prefetch && !prefetched_batch_weights);
    }
    return;
}

bool support_vulkan_expert(const ExpertPlan& expert,
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

static void admit_vulkan_expert(const CompiledModel& model,
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
        || !support_vulkan_expert(expert, *lease.gate_up, *lease.down, model.opt.optimization_flags))
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

ExpertVictimExecutionMetadata victim_metadata(const CompiledModel& model,
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
        || !support_vulkan_expert(expert,
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
static bool should_use_hybrid_expert_blocks(const CompiledModel& model,
                                            const MoeBlockPlan& moe,
                                            const LayerState& layer_state,
                                            ExecutionBackend backend) noexcept
{
    if (backend != ExecutionBackend::Vulkan || !model.expert_backend
        || layer_state.normalized.rows() < 32
        || layer_state.active_experts().empty())
    {
        return false;
    }

    size_t routed_rows = 0;
    for (const ExpertState& active : layer_state.active_experts())
    {
        if (active.batch.routes.empty()
            || active.batch.expert_id >= moe.experts.size())
        {
            return false;
        }
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        if (expert.gate_up_weight == invalid_tensor_handle
            || expert.down_weight == invalid_tensor_handle
            || !support_vulkan_expert(expert,
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

static Result<void> run_hybrid_expert_blocks(const CompiledModel& model,
                                             const MoeBlockPlan& moe,
                                             LayerState& layer_state,
                                             SessionStatistics& stats,
                                             ExpertScratch& scratch,
                                             uint32_t residency_group,
                                             bool prefetch)
{
    static constexpr uint32_t prefill_token_block_size = 32;
    const size_t active_expert_count = layer_state.active_experts().size();
    if (scratch.expert_workspaces.empty())
        scratch.expert_workspaces.resize(1);

    size_t total_routes = 0;
    for (const ExpertState& active : layer_state.active_experts())
        total_routes += active.batch.routes.size();

    std::vector<HybridBlockState> blocks;
    blocks.reserve(total_routes);
    scratch.backend_aggregated.assign(active_expert_count, 0);
    scratch.backend_aggregated_output_valid = false;
    scratch.backend_aggregated_output.reset(layer_state.normalized.rows(),
                                            model.descriptor.hidden_size,
                                            true);
    std::vector<ExpertBackendRequest> requests;
    requests.reserve(total_routes);
    std::vector<uint8_t> block_aggregated;
    std::vector<uint8_t> cpu_ready(active_expert_count, 0);

    for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
    {
        ExpertState& active = layer_state.active_experts()[active_index];
        size_t route_begin = 0;
        while (route_begin < active.batch.routes.size())
        {
            const size_t route_end = hybrid_block_end(active.batch.routes,
                                                      route_begin,
                                                      prefill_token_block_size);
            HybridBlockState block;
            block.active_index = active_index;
            block.input_begin = route_begin;
            gather_tokens(layer_state.normalized,
                          std::span<const ExpertRoute>(active.batch.routes).subspan(route_begin, route_end - route_begin),
                          block.input);
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
            const ExpertState& active = layer_state.active_experts()[active_index];
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
                const ExpertPlan& expert = moe.experts[layer_state.active_experts()[active_index].batch.expert_id];
                cache_requests.push_back({&model.weights.at(expert.gate_up_weight),
                                          &model.weights.at(expert.down_weight),
                                          residency_group,
                                          expert.cache_key,
                                          victim_metadata(model,
                                                          expert,
                                                          layer_state.normalized.rows())});
            }
            auto acquired = model.expert_cache->wait_acquire_ready_pairs(cache_requests,
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
                ExpertState& active = layer_state.active_experts()[active_index];
                active.lease = std::move(leases[lease_index]);
                cpu_ready[active_index] = 1;
            }
            active_indices.swap(remaining);
        }
        return {};
    };

    std::vector<size_t> pending_cpu;
    pending_cpu.reserve(active_expert_count);
    for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
    {
        ExpertState& active = layer_state.active_experts()[active_index];
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
            const ExpertPlan& expert = moe.experts[layer_state.active_experts()[active_index].batch.expert_id];
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
                const ExpertPlan& expert = moe.experts[layer_state.active_experts()[active_index].batch.expert_id];
                const auto wait_start = expert_cache_wait_start(model);
                auto lease = model.expert_cache->acquire_pair(model.weights.at(expert.gate_up_weight),
                                                              model.weights.at(expert.down_weight),
                                                              residency_group,
                                                              expert.cache_key,
                                                              victim_metadata(model,
                                                                              expert,
                                                                              layer_state.normalized.rows()));
                layer_state.active_experts()[active_index].metrics.cache_wait_time_microseconds += expert_cache_wait_elapsed(wait_start);
                if (!lease)
                    return lease.error();

                ExpertState& active = layer_state.active_experts()[active_index];
                active.lease = std::move(lease).value();
                const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                const TensorData& down = model.weights.at(expert.down_weight);
                const bool mapped_bfloat16 = gate_up.dtype == DType::BFloat16
                                             && down.dtype == DType::BFloat16
                                             && gate_up.mapped_data != nullptr
                                             && down.mapped_data != nullptr;
                cpu_ready[active_index] = mapped_bfloat16 ? 1 : 0;
                admit_vulkan_expert(model,
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
        const ExpertState& active = layer_state.active_experts()[active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        admit_vulkan_expert(model,
                            expert,
                            active.lease,
                            residency_group,
                            static_cast<uint32_t>(layer_state.normalized.rows()),
                            ExecutionBackend::Vulkan);
    }
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    {
        HybridBlockState& block = blocks[block_index];
        const ExpertState& active = layer_state.active_experts()[block.active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        ExpertBackendRequest request{
            expert.cache_key,
            &block.input,
            &block.output,
            expert.weight_size,
            {}};
        request.route_aggregation.output = &scratch.backend_aggregated_output;
        request.route_aggregation.routes = std::span<const ExpertRoute>(active.batch.routes).subspan(block.input_begin, block.input.rows());
        request.route_aggregation.token_count = static_cast<uint32_t>(layer_state.normalized.rows());
        request.route_aggregation.completed = &block_aggregated[block_index];
        request.route_aggregation.require_all_requests = true;
        request.device_input = layer_state.normalized_device;
        request.device_routes = request.route_aggregation.routes;
        requests.push_back(request);
    }

    std::unique_ptr<ExpertSubmission> submission = model.expert_backend->submit_batch(requests);
    if (!submission)
        return Error{ErrorCode::InternalError, "hybrid expert block submission failed"};

    const std::span<const ExpertBackendExecutionResult> planned = submission->reservations();
    if (planned.size() != requests.size())
    {
        submission->abort();
        return Error{ErrorCode::InternalError, "hybrid expert block reservation shape mismatch"};
    }
    for (size_t request_index = 0; request_index < planned.size(); ++request_index)
    {
        HybridBlockState& block = blocks[request_index];
        if (planned[request_index] == ExpertBackendExecutionResult::Executed)
        {
            block.gpu_planned = true;
        }
        else
        {
            add_cpu_work(block.active_index);
        }
    }

    const auto run_pending_cpu_blocks = [&]() -> Result<void> {
        size_t block_begin = 0;
        for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
        {
            // Blocks are built in Expert order; visit only this Expert's range.
            size_t block_end = block_begin;
            while (block_end < blocks.size() && blocks[block_end].active_index == active_index)
                ++block_end;
            const std::span<HybridBlockState> expert_blocks = std::span<HybridBlockState>(blocks).subspan(block_begin, block_end - block_begin);
            block_begin = block_end;
            size_t pending_before = 0;
            for (const HybridBlockState& block : expert_blocks)
            {
                if (!block.gpu_planned && !block.cpu_executed)
                {
                    ++pending_before;
                }
            }
            if (pending_before == 0)
                continue;

            if (cpu_ready[active_index] != 1)
            {
                auto acquired = acquire_cpu_weights({active_index});
                if (!acquired)
                    return acquired.error();
            }
            run_hybrid_cpu_blocks(model,
                                  moe,
                                  layer_state,
                                  scratch.expert_workspaces.front(),
                                  expert_blocks,
                                  active_index,
                                  prefetch);
            size_t pending_after = 0;
            for (const HybridBlockState& block : expert_blocks)
            {
                if (!block.gpu_planned && !block.cpu_executed)
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
            layer_state.active_experts()[active_index].lease = {};
            cpu_ready[active_index] = 0;
        }
        return {};
    };

    auto initial_cpu = run_pending_cpu_blocks();
    if (!initial_cpu)
        return initial_cpu.error();

    std::vector<ExpertBackendExecutionResult>& results = scratch.backend_results;
    results.clear();
    submission->wait(results);
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
        }
        else
        {
            block.gpu_planned = false;
            add_cpu_work(block.active_index);
        }
    }

    std::fill(scratch.backend_aggregated.begin(),
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

    for (const HybridBlockState& block : blocks)
    {
        if (!block.gpu_executed && !block.cpu_executed)
            return Error{ErrorCode::InternalError, "hybrid expert block produced no output"};
    }
    assemble_hybrid_outputs(layer_state,
                            blocks,
                            std::span<const uint8_t>(scratch.backend_aggregated));

    // Let cold GPU admissions overlap the CPU fallback for this layer, then
    // finish them before advancing so the warmed entries are available to
    // later decode passes.
    model.expert_backend->wait_for_background_work();

    for (ExpertState& active : layer_state.active_experts())
        active.lease = {};

    stats.expert_batches += active_expert_count;
    if (expert_cache_wait_timing_enabled(model))
    {
        for (const ExpertState& active : layer_state.active_experts())
        {
            for (const ExpertRoute& route : active.batch.routes)
            {
                if (route.rank >= maximum_expert_route_ranks)
                    continue;
                ++stats.expert_route_rank_demands[route.rank];
                stats.expert_route_rank_demand_queue_time_microseconds[route.rank] += active.metrics.cache_wait_time_microseconds;
            }
        }
    }
    layer_state.experts_executed = true;
    return {};
}

void prepare_experts(const MoeBlockPlan& moe,
                     LayerState& layer_state,
                     SessionStatistics& stats)
{
    ExpertDispatchPlan& plan = layer_state.dispatch_plan;
    stats.expert_assignments += plan.assignment_count;
    layer_state.resize_experts(plan.batches.size());
    for (size_t batch_index = 0; batch_index < plan.batches.size(); ++batch_index)
    {
        ExpertBatch& batch = plan.batches[batch_index];
        stats.expert_token_counts[batch.expert_id] += batch.routes.size();
        const uint64_t weight_size = moe.experts[batch.expert_id].weight_size;
        if (stats.expert_batch_weight_bytes > std::numeric_limits<uint64_t>::max() - weight_size)
            stats.expert_batch_weight_bytes = std::numeric_limits<uint64_t>::max();
        else
            stats.expert_batch_weight_bytes += weight_size;
        const uint64_t route_count = static_cast<uint64_t>(batch.routes.size());
        const uint64_t route_weight_size = weight_size != 0
                                                   && batch.routes.size() > std::numeric_limits<uint64_t>::max() / weight_size
                                               ? std::numeric_limits<uint64_t>::max()
                                               : weight_size * route_count;
        if (stats.expert_route_weight_bytes > std::numeric_limits<uint64_t>::max() - route_weight_size)
            stats.expert_route_weight_bytes = std::numeric_limits<uint64_t>::max();
        else
            stats.expert_route_weight_bytes += route_weight_size;
        layer_state.active_experts()[batch_index].prepare(batch);
    }
}

Result<void> request_experts(const CompiledModel& model,
                             const MoeBlockPlan& moe,
                             const LayerState& layer_state,
                             ExpertScratch& scratch,
                             uint32_t residency_group)
{
    if (!model.expert_cache)
        return {};
    std::vector<ExpertCachePairRequest>& requests = scratch.cache_requests;
    requests.clear();
    requests.reserve(layer_state.active_experts().size());
    for (const ExpertState& active : layer_state.active_experts())
    {
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        if (expert.gate_up_weight == invalid_tensor_handle || expert.cache_key.empty())
            continue;
        requests.push_back({
            &model.weights.at(expert.gate_up_weight),
            &model.weights.at(expert.down_weight),
            residency_group,
            expert.cache_key,
            victim_metadata(model, expert, layer_state.normalized.rows()),
        });
    }
    auto requested = model.expert_cache->request_pairs(requests);
    if (!requested)
        return requested.error();
    return {};
}

Result<void> forward_moe(const CompiledModel& model,
                         const MoeBlockPlan& moe,
                         LayerState& layer_state,
                         SessionStatistics& stats,
                         ExpertScratch& scratch,
                         uint32_t residency_group,
                         ExecutionBackend backend,
                         bool prefetch,
                         const std::function<void()>& independent_work)
{
    // Declare before submission owners so exception/early-return cleanup
    // releases retained weights only after in-flight work has been drained.
    struct WaveCleanup
    {
        ExpertScratch& scratch;
        ~WaveCleanup()
        {
            scratch.backend_requests.clear();
            scratch.prestaged_demands.clear();
            scratch.prestaged_pins.clear();
            scratch.prestaged_cache_leases.clear();
            scratch.current_wave_requests.clear();
            scratch.next_wave_requests.clear();
            scratch.wave_demands.clear();
            scratch.prepared_pins.clear();
        }
    } wave_cleanup{scratch};
    bool independent_work_started = false;
    const auto run_independent_work = [&] {
        if (!independent_work || independent_work_started) return;
        independent_work_started = true;
        independent_work();
    };
    const bool gpu_priority = backend == ExecutionBackend::Vulkan
                              && model.opt.hybrid_mode == HybridMode::HybridExperts
                              && has_flag(model.opt.optimization_flags, OptimizationVulkanExpertGpuPriority)
                              && !has_flag(model.opt.flags, OptionDisableGpuExpertExecution);
    if (!gpu_priority && should_use_hybrid_expert_blocks(model, moe, layer_state, backend))
        return run_hybrid_expert_blocks(model,
                                        moe,
                                        layer_state,
                                        stats,
                                        scratch,
                                        residency_group,
                                        prefetch);

    const size_t active_expert_count = layer_state.active_experts().size();
    const bool gpu_expert_batch_eligible = layer_state.normalized.rows() >= vulkan_expert_gpu_min_rows
                                           || model.opt.hybrid_mode == HybridMode::HybridExperts;
    const bool use_backend_batch = backend == ExecutionBackend::Vulkan
                                   && model.expert_backend
                                   && gpu_expert_batch_eligible
                                   && !has_flag(model.opt.flags, OptionDisableGpuExpertExecution);
    const auto can_use_device_input = [&](const ExpertState& active) {
        if (!use_backend_batch || !layer_state.normalized_device || layer_state.normalized_device->empty()) return false;
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        if (expert.gate_up_weight == invalid_tensor_handle || expert.down_weight == invalid_tensor_handle) return false;
        const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
        const TensorData& down = model.weights.at(expert.down_weight);
        return gate_up.dtype == DType::MxFp4 && down.dtype == DType::MxFp4
               && support_vulkan_expert(expert, gate_up, down, model.opt.optimization_flags);
    };
#if defined(_OPENMP)
    int expert_team_size = 1;
    bool parallelize_regroup = false;
    expert_team_size = std::min(static_cast<int>(active_expert_count), static_cast<int>(cpu_linear_num_threads()));
    if (expert_team_size > 1)
    {
        uint64_t regroup_element_count = 0;
        for (const ExpertState& active : layer_state.active_experts())
        {
            if (!can_use_device_input(active))
                regroup_element_count += static_cast<uint64_t>(active.batch.routes.size())
                                         * layer_state.normalized.columns();
        }
        static constexpr uint64_t minimum_parallel_regroup_elements = 256 * 1024;
        parallelize_regroup = regroup_element_count >= minimum_parallel_regroup_elements;
    }
#endif
    const int64_t parallel_expert_count = static_cast<int64_t>(active_expert_count);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(expert_team_size) if (parallelize_regroup)
#endif
    for (int64_t expert_index = 0; expert_index < parallel_expert_count; ++expert_index)
    {
        ExpertState& active = layer_state.active_experts()[static_cast<size_t>(expert_index)];
        if (can_use_device_input(active))
        {
            // clear() retains capacity but prevents stale rows from being
            // mistaken for this pass's fallback input.
            active.input.clear();
            continue;
        }
        gather_tokens(layer_state.normalized, active.batch.routes, active.input);
    }

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
    backend_executed.clear();
    backend_aggregated.clear();
    backend_indices.clear();
    backend_requests.clear();
    scratch.backend_aggregated_output_valid = false;
    std::unique_ptr<ExpertSubmission> backend_submission;
    bool backend_reserved_work = false;
    bool backend_reservation_shape_valid = true;
    if (use_backend_batch)
    {
        backend_executed.assign(active_expert_count, 0);
        backend_aggregated.assign(active_expert_count, 0);
        backend_indices.reserve(active_expert_count);
        backend_requests.reserve(active_expert_count);
        for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
        {
            ExpertState& active = layer_state.active_experts()[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            if (expert.gate_up_weight == invalid_tensor_handle
                || expert.down_weight == invalid_tensor_handle
                || !support_vulkan_expert(expert,
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
                model.expert_backend->admit(expert.cache_key,
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
            ExpertBackendRequest request{expert.cache_key, active.input.rows() ? &active.input : nullptr, &active.output, expert.weight_size, {}};
            request.input_rows = active.batch.routes.size();
            request.input_columns = layer_state.normalized.columns();
            request.route_aggregation.output = &scratch.backend_aggregated_output;
            request.route_aggregation.routes = active.batch.routes;
            request.route_aggregation.token_count = static_cast<uint32_t>(layer_state.normalized.rows());
            request.route_aggregation.completed = &backend_aggregated[active_index];
            request.device_input = layer_state.normalized_device;
            request.device_routes = active.batch.routes;
            if (layer_state.normalized_device && gate_up_weight.dtype == DType::MxFp4
                && down_weight.dtype == DType::MxFp4)
                request.device_output = &active.device_output;
            backend_requests.push_back(request);
        }
        const bool complete_backend_coverage = backend_requests.size() == active_expert_count;
        if (complete_backend_coverage && !backend_requests.empty())
            scratch.backend_aggregated_output.reset(layer_state.normalized.rows(), model.descriptor.hidden_size, true);
        for (ExpertBackendRequest& request : backend_requests)
        {
            if (complete_backend_coverage)
                request.route_aggregation.require_all_requests = true;
            else
                request.route_aggregation = {};
        }
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
                    backend_reserved_work = true;
                }
            }
        }
    }

    const auto finalize_backend = [&]() -> Result<void> {
        if (backend_submission)
        {
            std::vector<ExpertBackendExecutionResult>& backend_results = scratch.backend_results;
            backend_results.clear();
            backend_submission->wait(backend_results);
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
            scratch.backend_aggregated_output_valid = std::any_of(backend_aggregated.begin(),
                                                                  backend_aggregated.end(),
                                                                  [](uint8_t value) { return value != 0; });
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
                    failed_indices.push_back(active_index);
                    continue;
                }
            }

            if (!gpu_priority && !failed_indices.empty())
            {
                for (size_t active_index : failed_indices)
                {
                    ExpertState& active = layer_state.active_experts()[active_index];
                    active.device_output.reset();
                    const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                    const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                    const TensorData& down = model.weights.at(expert.down_weight);
                    if (model.expert_cache && !expert.cache_key.empty())
                    {
                        const auto wait_start = expert_cache_wait_start(model);
                        auto lease = model.expert_cache->acquire_pair(gate_up, down, residency_group, expert.cache_key,
                                                                      victim_metadata(model, expert, layer_state.normalized.rows()));
                        active.metrics.cache_wait_time_microseconds += expert_cache_wait_elapsed(wait_start);
                        if (!lease) return lease.error();
                        active.lease = std::move(lease).value();
                        admit_vulkan_expert(model, expert, active.lease, residency_group,
                                            static_cast<uint32_t>(layer_state.normalized.rows()), backend);
                    }
                }
                forward_experts(model, moe, layer_state, failed_indices, scratch, prefetch);
                for (size_t active_index : failed_indices) layer_state.active_experts()[active_index].lease = {};
            }
            backend_submission.reset();
        }

        return {};
    };
    std::vector<size_t>& prestaged_indices = scratch.prestaged_indices;
    std::vector<ExpertDemandRequest>& prestaged_demands = scratch.prestaged_demands;
    std::vector<std::shared_ptr<const void>>& prestaged_pins = scratch.prestaged_pins;
    prestaged_indices.clear();
    prestaged_demands.clear();
    prestaged_pins.clear();
    std::unique_ptr<ExpertDemandSubmission> prestaged_upload;
    size_t prestaged_prepared = 0;
    if (gpu_priority && backend_reserved_work && backend_reservation_shape_valid)
    {
        // The resident batch can compute while a first already-ready cold
        // prefix uploads. Do not wait for host reads here: first completion
        // selection remains the responsibility of the normal pending loop.
        std::vector<size_t>& candidates = scratch.prestaged_candidates;
        candidates.clear();
        candidates.reserve(8);
        for (size_t active_index : backend_indices)
        {
            const auto& expert = moe.experts[layer_state.active_experts()[active_index].batch.expert_id];
            if (!backend_executed[active_index] && !expert.cache_key.empty())
                candidates.push_back(active_index);
            if (candidates.size() == 8) break;
        }
        if (model.expert_cache && !candidates.empty())
        {
            std::vector<ExpertCachePairRequest>& requests = scratch.prestaged_cache_requests;
            std::vector<ExpertCacheLease>& leases = scratch.prestaged_cache_leases;
            requests.clear();
            leases.clear();
            leases.resize(candidates.size());
            requests.reserve(candidates.size());
            for (size_t active_index : candidates)
            {
                const auto& expert = moe.experts[layer_state.active_experts()[active_index].batch.expert_id];
                requests.push_back({&model.weights.at(expert.gate_up_weight), &model.weights.at(expert.down_weight),
                                    residency_group, expert.cache_key,
                                    victim_metadata(model, expert, layer_state.normalized.rows())});
            }
            auto ready = model.expert_cache->wait_acquire_ready_pairs(requests, leases, true, 1, false);
            if (!ready) return ready.error();
            for (size_t index = 0; index < candidates.size(); ++index)
            {
                if (!leases[index].gate_up || !leases[index].down) continue;
                layer_state.active_experts()[candidates[index]].lease = std::move(leases[index]);
                prestaged_indices.push_back(candidates[index]);
            }
        }
        else
        {
            prestaged_indices.swap(candidates);
        }
        for (size_t active_index : prestaged_indices)
        {
            const auto& active = layer_state.active_experts()[active_index];
            const auto& expert = moe.experts[active.batch.expert_id];
            prestaged_demands.push_back({expert.cache_key,
                                         active.lease.gate_up ? active.lease.gate_up : borrow_resident_tensor(model.weights.at(expert.gate_up_weight)),
                                         expert.gate_up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_bias),
                                         active.lease.down ? active.lease.down : borrow_resident_tensor(model.weights.at(expert.down_weight)),
                                         expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias),
                                         residency_group, expert.activation_limit, expert.activation});
        }
        if (!prestaged_demands.empty())
        {
            prestaged_upload = model.expert_backend->begin_demand_batch(prestaged_demands);
            // Shared computation depends only on normalized input. Start it
            // after cold transfer work exists, while resident requests and
            // the upload still own their independent bounded reservations.
            if (prestaged_upload) run_independent_work();
        }
    }
    if (gpu_priority)
    {
        // Resolve failed resident reservations before classifying pending
        // work. The first cold upload owns independent bounded host/device
        // references and must not make those failed requests disappear.
        auto finalized = finalize_backend();
        if (!finalized) return finalized.error();
    }
    if (prestaged_upload)
    {
        prestaged_pins.resize(prestaged_demands.size());
        prestaged_prepared = prestaged_upload->wait(prestaged_pins);
        bool valid = prestaged_prepared <= prestaged_pins.size();
        if (valid)
            for (size_t index = 0; index < prestaged_prepared; ++index) valid = valid && static_cast<bool>(prestaged_pins[index]);
        if (!valid)
        {
            prestaged_upload->abort();
            prestaged_prepared = 0;
        }
        for (size_t index = prestaged_prepared; index < prestaged_pins.size(); ++index) prestaged_pins[index].reset();
        prestaged_upload.reset();
    }

    for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
    {
        ExpertState& active = layer_state.active_experts()[active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        if (use_backend_batch && backend_executed[active_index])
            continue;
        if (std::find(prestaged_indices.begin(), prestaged_indices.end(), active_index) != prestaged_indices.end())
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
    if (!pending.empty() && !gpu_priority)
    {
        std::vector<ExpertCachePairRequest>& requests = scratch.cache_requests;
        std::vector<ExpertCacheLease>& leases = scratch.cache_leases;
        requests.clear();
        leases.clear();
        leases.resize(pending.size());
        requests.reserve(pending.size());
        for (size_t active_index : pending)
        {
            const ExpertPlan& expert = moe.experts[layer_state.active_experts()[active_index].batch.expert_id];
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
                ExpertState& active = layer_state.active_experts()[pending[pending_index]];
                active.lease = std::move(leases[pending_index]);
                const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                admit_vulkan_expert(model,
                                    expert,
                                    active.lease,
                                    residency_group,
                                    static_cast<uint32_t>(layer_state.normalized.rows()),
                                    backend);
            }
        }
    }

    // Keep exactly one compute wave in flight while the next bounded demand
    // upload is prepared. This state spans first-ready host-cache batches so
    // a single ready Expert does not force a compute fence before the next
    // read or upload can begin.
    std::vector<size_t>& current_wave_indices = scratch.current_wave_indices;
    std::vector<ExpertBackendRequest>& current_wave_requests = scratch.current_wave_requests;
    current_wave_indices.clear();
    current_wave_requests.clear();
    std::unique_ptr<ExpertSubmission> current_wave_submission;
    const auto finish_current_wave = [&] {
        if (current_wave_requests.empty()) return;
        std::vector<ExpertBackendExecutionResult>& results = scratch.backend_results;
        results.clear();
        bool valid = false;
        if (current_wave_submission)
        {
            const auto planned = current_wave_submission->reservations();
            valid = planned.size() == current_wave_requests.size();
            current_wave_submission->wait(results);
            valid = valid && results.size() == current_wave_requests.size();
            if (valid)
            {
                for (size_t index = 0; index < results.size(); ++index)
                    if (results[index] == ExpertBackendExecutionResult::Executed
                        && planned[index] != ExpertBackendExecutionResult::Executed) valid = false;
            }
            valid = valid && current_wave_submission->commit();
            if (!valid) current_wave_submission->abort();
        }
        std::vector<size_t>& failed_indices = scratch.wave_failed_indices;
        failed_indices.clear();
        failed_indices.reserve(current_wave_indices.size());
        for (size_t index = 0; index < current_wave_indices.size(); ++index)
        {
            const size_t active_index = current_wave_indices[index];
            auto& active = layer_state.active_experts()[active_index];
            if (valid && results[index] == ExpertBackendExecutionResult::Executed)
            {
                backend_executed[active_index] = 1;
            }
            else
            {
                active.device_output.reset();
                failed_indices.push_back(active_index);
            }
        }
        // Release device Entry pins before a capacity-limited upload retries.
        current_wave_submission.reset();
        forward_experts(model, moe, layer_state, failed_indices, scratch, prefetch);
        for (size_t active_index : current_wave_indices)
            layer_state.active_experts()[active_index].lease = {};
        current_wave_indices.clear();
        current_wave_requests.clear();
    };

    const auto forward_ready = [&](const std::vector<size_t>& indices,
                                   std::span<std::shared_ptr<const void>> preload_pins = {},
                                   size_t preload_prepared = 0) {
        std::vector<size_t>& cpu_indices = scratch.wave_cpu_indices;
        cpu_indices.clear();
        cpu_indices.reserve(indices.size());
        std::vector<ExpertDemandRequest>& demands = scratch.wave_demands;
        std::vector<size_t>& demand_indices = scratch.wave_demand_indices;
        demands.clear();
        demand_indices.clear();
        demands.reserve(indices.size());
        demand_indices.reserve(indices.size());
        for (size_t active_index : indices)
        {
            ExpertState& active = layer_state.active_experts()[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            bool supported = false;
            if (gpu_priority && use_backend_batch && !expert.cache_key.empty()
                && expert.gate_up_weight != invalid_tensor_handle && expert.down_weight != invalid_tensor_handle)
            {
                const auto gate_up = active.lease.gate_up ? active.lease.gate_up : borrow_resident_tensor(model.weights.at(expert.gate_up_weight));
                const auto down = active.lease.down ? active.lease.down : borrow_resident_tensor(model.weights.at(expert.down_weight));
                supported = support_vulkan_expert(expert, *gate_up, *down, model.opt.optimization_flags);
                if (supported)
                {
                    demands.push_back({expert.cache_key, gate_up,
                                       expert.gate_up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_bias),
                                       down, expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias),
                                       residency_group, expert.activation_limit, expert.activation});
                    demand_indices.push_back(active_index);
                }
            }
            if (!supported)
            {
                active.device_output.reset();
                cpu_indices.push_back(active_index);
            }
        }
        static constexpr size_t maximum_demand_wave = 8;
        size_t cursor = 0;
        while (cursor < demands.size())
        {
            const size_t attempt_count = std::min(demands.size() - cursor, maximum_demand_wave);
            std::vector<std::shared_ptr<const void>>& prepared_pins = scratch.prepared_pins;
            prepared_pins.clear();
            prepared_pins.resize(attempt_count);
            const auto wave_demands = std::span<const ExpertDemandRequest>(demands).subspan(cursor, attempt_count);
            // Independent transfer recording/submission can run concurrently
            // with the preceding compute command buffer. wait() publishes only
            // completed weights; no unfinished VkMat reaches submit_batch.
            std::unique_ptr<ExpertDemandSubmission> upload;
            size_t prepared = 0;
            if (cursor == 0 && preload_prepared != 0 && preload_pins.size() == attempt_count)
            {
                prepared = preload_prepared;
                for (size_t index = 0; index < prepared_pins.size(); ++index)
                    prepared_pins[index] = std::move(preload_pins[index]);
                preload_prepared = 0;
            }
            else
            {
                // A zero preloaded prefix (including transient capacity
                // pressure from the initial resident batch) retries normally
                // after that batch's reservation has been released.
                for (auto& pin : preload_pins) pin.reset();
                upload = model.expert_backend->begin_demand_batch(wave_demands);
                // Do not move Shared ahead of cold demand: its compute CB
                // overlaps the already-started upload and preparation lane.
                if (upload) run_independent_work();
                prepared = upload ? upload->wait(prepared_pins) : 0;
            }
            bool valid_prefix = prepared <= attempt_count;
            if (valid_prefix)
                for (size_t index = 0; index < prepared; ++index) valid_prefix = valid_prefix && static_cast<bool>(prepared_pins[index]);
            if (!valid_prefix && upload) upload->abort();
            upload.reset();
            if (!valid_prefix || prepared == 0)
            {
                prepared_pins.clear();
                if (!current_wave_requests.empty())
                {
                    finish_current_wave();
                    continue; // Retry after both current host/device pins release.
                }
                const size_t active_index = demand_indices[cursor++];
                layer_state.active_experts()[active_index].device_output.reset();
                cpu_indices.push_back(active_index);
                continue;
            }
            std::vector<size_t>& next_indices = scratch.next_wave_indices;
            std::vector<ExpertBackendRequest>& next_requests = scratch.next_wave_requests;
            next_indices.clear();
            next_requests.clear();
            next_indices.reserve(prepared);
            next_requests.reserve(prepared);
            for (size_t index = 0; index < prepared; ++index)
            {
                const size_t active_index = demand_indices[cursor + index];
                ExpertState& active = layer_state.active_experts()[active_index];
                const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                const auto& demand = demands[cursor + index];
                ExpertBackendRequest request{expert.cache_key, active.input.rows() ? &active.input : nullptr, &active.output, expert.weight_size, {}};
                request.input_rows = active.batch.routes.size();
                request.input_columns = layer_state.normalized.columns();
                request.device_input = layer_state.normalized_device;
                request.device_routes = active.batch.routes;
                if (layer_state.normalized_device && demand.gate_up->dtype == DType::MxFp4 && demand.down->dtype == DType::MxFp4)
                    request.device_output = &active.device_output;
                next_indices.push_back(active_index);
                next_requests.push_back(request);
            }
            // An upload-ready next wave cannot be lost while the preceding
            // wave commits: its bounded pins protect every selected pair.
            finish_current_wave();
            current_wave_indices.swap(next_indices);
            current_wave_requests.swap(next_requests);
            current_wave_submission = model.expert_backend->submit_batch(current_wave_requests);
            prepared_pins.clear(); // submit_batch owns the selected Entry pins.
            cursor += prepared;
        }
        forward_experts(model, moe, layer_state, cpu_indices, scratch, prefetch);
        for (size_t active_index : cpu_indices)
            layer_state.active_experts()[active_index].lease = {};
        // Keep bookkeeping capacity, but release the completed upload's Host
        // weights before waiting for the next host-cache batch. Only current
        // compute leases may remain pinned across that capacity-limited wait.
        demands.clear();
    };

    const auto preload_pins = std::span<std::shared_ptr<const void>>(prestaged_pins);
    forward_ready(prestaged_indices, preload_pins, prestaged_prepared);
    prestaged_pins.clear();
    prestaged_demands.clear();
    forward_ready(uncached);

    if (ready_batch_acquired)
    {
        forward_ready(pending);
        for (size_t active_index : pending)
            layer_state.active_experts()[active_index].lease = {};
        pending.clear();
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
        // Coalesce at most one bounded GPU wave. Never wait on a tail that
        // cannot be admitted while this prefix owns host-cache capacity.
        const size_t requested_count = gpu_priority ? std::min(pending.size(), size_t(8)) : pending.size();
        requests.reserve(requested_count);
        leases.resize(requested_count);
        for (size_t active_index : std::span<const size_t>(pending).first(requested_count))
        {
            const ExpertPlan& expert = moe.experts[layer_state.active_experts()[active_index].batch.expert_id];
            requests.push_back({
                &model.weights.at(expert.gate_up_weight),
                &model.weights.at(expert.down_weight),
                residency_group,
                expert.cache_key,
                victim_metadata(model, expert, layer_state.normalized.rows()),
            });
        }

        const auto cache_wait_start = expert_cache_wait_start(model);
        // Poll next host-ready work while current compute owns its leases.
        // A small cache may be entirely pinned by that wave; release it before
        // entering a blocking host wait so capacity pressure cannot deadlock.
        auto acquired = model.expert_cache->wait_acquire_ready_pairs(requests, leases, true, 1,
                                                                     current_wave_requests.empty());
        if (acquired && acquired.value() == 0 && !current_wave_requests.empty())
        {
            auto queued = model.expert_cache->request_pairs(requests);
            if (!queued) return queued.error();
            if (queued.value() == 0)
                finish_current_wave();
            // A nonzero admitted prefix has a loading/ready pair independent
            // of the current leases. Its first read can complete alongside
            // current GPU work without waiting for the rest of the wave.
            acquired = model.expert_cache->wait_acquire_ready_pairs(requests, leases, true);
        }
        const uint64_t cache_wait_microseconds = expert_cache_wait_elapsed(cache_wait_start);
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
            if (pending_index >= requested_count)
            {
                pending[pending_count++] = pending[pending_index];
                continue;
            }
            ExpertCacheLease& lease = leases[pending_index];
            if (!lease.gate_up)
            {
                pending[pending_count++] = pending[pending_index];
                continue;
            }
            const size_t active_index = pending[pending_index];
            ExpertState& active = layer_state.active_experts()[active_index];
            active.lease = std::move(lease);
            if (!wait_accounted)
            {
                active.metrics.cache_wait_time_microseconds += cache_wait_microseconds;
                wait_accounted = true;
            }
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            admit_vulkan_expert(model,
                                expert,
                                active.lease,
                                residency_group,
                                static_cast<uint32_t>(layer_state.normalized.rows()),
                                backend);
            ready_indices.push_back(active_index);
        }
        pending.resize(pending_count);
        forward_ready(ready_indices);
    }

    finish_current_wave();

    if (!gpu_priority)
    {
        auto finalized = finalize_backend();
        if (!finalized) return finalized.error();
    }

    stats.expert_batches += active_expert_count;
    if (expert_cache_wait_timing_enabled(model))
    {
        for (const ExpertState& active : layer_state.active_experts())
        {
            for (const ExpertRoute& route : active.batch.routes)
            {
                if (route.rank >= maximum_expert_route_ranks)
                    continue;
                ++stats.expert_route_rank_demands[route.rank];
                stats.expert_route_rank_demand_queue_time_microseconds[route.rank] += active.metrics.cache_wait_time_microseconds;
            }
        }
    }
    layer_state.experts_executed = true;
    return {};
}

bool init_moe_output(ExpertScratch& scratch,
                     size_t rows,
                     uint32_t columns,
                     ActivationBuffer& output)
{
    if (!scratch.backend_aggregated_output_valid
        || scratch.backend_aggregated_output.rows() != rows
        || scratch.backend_aggregated_output.columns() != columns)
    {
        scratch.backend_aggregated_output_valid = false;
        output.reset(rows, columns, true);
        return false;
    }

    // The committed aggregate becomes Combine's accumulator; return the old
    // normalized storage for the next backend submission.
    output.swap(scratch.backend_aggregated_output);
    scratch.backend_aggregated_output_valid = false;
    return true;
}

} // namespace moe
} // namespace ncnn
