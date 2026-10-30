#include "expert.h"

#include "sessionstate.h"
#include "expertbackend.h"
#include "graph/compiledmodel.h"
#include "kernels/bfloat16.h"
#include "kernels/fastmath.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"
#include "kernels/int4.h"
#include "kernels/vector.h"
#include "storage/expertcache.h"
#include "storage/weightstore.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/modelpipeline.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
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

static float gated_product(float gate, float up, ExpertActivation activation,
                           float limit, uint64_t flags)
{
    if (activation == ExpertActivation::GptOssSwiGlu)
    {
        if (limit > 0.0f)
        {
            gate = std::min(gate, limit);
            up = std::clamp(up, -limit, limit);
        }
        return scaled_silu(gate, 1.702f, flags) * (up + 1.0f);
    }
    if (activation == ExpertActivation::DeepSeekSwiGlu && limit > 0.0f)
        up = std::clamp(up, -limit, limit);
    return activate(gate, activation, limit, flags) * up;
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
        if (gate_up_weight.dtype == DType::MxFp4 && gate_up_weight.input_column_indices.empty())
        {
            forward_gate_up_mxfp4(gate_up_weight,
                                  gate_up_bias,
                                  input,
                                  expert.activation,
                                  expert.activation_limit,
                                  activated,
                                  optimization_flags,
                                  &workspace.gate);
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
                    destination[column] = gated_product(gate, up, expert.activation,
                                                        expert.activation_limit, optimization_flags);
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
                    destination[column] = gated_product(source[column * 2], source[column * 2 + 1],
                                                        expert.activation, expert.activation_limit,
                                                        optimization_flags);
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
            && up_weight.dtype == DType::Float8E4M3
            && expert.gate_bias == invalid_tensor_handle
            && expert.up_bias == invalid_tensor_handle)
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
    if (prefetch && !gate_prefetched && expert.layout != ExpertLayout::UpDown)
        prefetch_weight(weights, expert.gate_weight);
    const bool paired = expert.layout != ExpertLayout::UpDown
                        && expert.gate_bias == invalid_tensor_handle
                        && expert.up_bias == invalid_tensor_handle
                        && (up_weight.dtype == DType::BFloat16 || up_weight.dtype == DType::Float8E4M3
#if defined(__aarch64__) || defined(_M_ARM64)
                            || is_qnk_dtype(up_weight.dtype)
#endif
                                )
                        && forward_linear({&up_weight, &weights.at(expert.gate_weight)}, input,
                                          {&up, &workspace.gate}, optimization_flags,
                                          {operators.find_weight(expert.up_weight),
                                           operators.find_weight(expert.gate_weight)},
                                          &workspace.quantized_input);
    if (!paired)
        expert_linear(up_weight,
                      expert.up_bias == invalid_tensor_handle ? nullptr : &weights.at(expert.up_bias),
                      operators.find_weight(expert.up_weight),
                      input,
                      up,
                      optimization_flags,
                      &workspace.quantized_input);
    if (expert.layout != ExpertLayout::UpDown)
    {
        ActivationBuffer& gate = workspace.gate;
        if (!paired)
            expert_linear(weights.at(expert.gate_weight),
                          expert.gate_bias == invalid_tensor_handle ? nullptr : &weights.at(expert.gate_bias),
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
                up_row[column] = gated_product(gate_row[column], up_row[column], expert.activation,
                                               expert.activation_limit, optimization_flags);
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
                  expert.down_bias == invalid_tensor_handle ? nullptr : &weights.at(expert.down_bias),
                  operators.find_weight(expert.down_weight),
                  up,
                  output,
                  optimization_flags,
                  &workspace.quantized_input);
}

// The model owns these tensors until its backend has drained all uploads.
// Borrowing avoids a host copy while the asynchronous demand retains them.
static std::shared_ptr<const TensorData> borrow_resident_tensor(const TensorData& tensor)
{
    return std::shared_ptr<const TensorData>(std::shared_ptr<const TensorData>{}, &tensor);
}

static bool forward_cached_dense_ffn(const CompiledModel& model,
                                     const ExpertPlan& ffn,
                                     const ActivationBuffer& input,
                                     ActivationBuffer& output)
{
    if (!model.expert_backend || ffn.cache_key.empty()
        || model.opt.hybrid_mode != HybridMode::HybridExperts
        || has_flag(model.opt.flags, OptionDisableGpuExpertExecution))
        return false;

    ExpertRequest request;
    request.key = ffn.cache_key;
    request.input = &input;
    request.output = &output;
    request.weight_size = ffn.weight_size;
    const auto execute = [&]() {
        auto submission = model.expert_backend->submit_batch({&request, 1});
        if (!submission)
            return ExpertStatus::Failed;
        const auto reservations = submission->reservations();
        if (reservations.size() != 1)
        {
            submission->abort();
            return ExpertStatus::Failed;
        }
        if (reservations.front() != ExpertStatus::Executed)
        {
            const auto result = reservations.front();
            submission->abort();
            return result;
        }
        std::vector<ExpertStatus> results;
        submission->wait(results);
        if (results.size() != 1 || results.front() != ExpertStatus::Executed
            || !submission->commit())
        {
            submission->abort();
            return ExpertStatus::Failed;
        }
        return output.rows() == input.rows()
                       && output.columns() == model.weights.at(ffn.down_weight).shape[0]
                   ? ExpertStatus::Executed
                   : ExpertStatus::Failed;
    };
    const ExpertStatus resident = execute();
    if (resident == ExpertStatus::Executed)
        return true;
    if (resident != ExpertStatus::NotResident)
        return false;

    const TensorHandle gate = ffn.gate_up_weight == invalid_tensor_handle ? ffn.gate_weight : ffn.gate_up_weight;
    const TensorHandle gate_bias = ffn.gate_up_weight == invalid_tensor_handle ? ffn.gate_bias : ffn.gate_up_bias;
    ExpertDemandRequest demand;
    demand.key = ffn.cache_key;
    demand.gate_up = borrow_resident_tensor(model.weights.at(gate));
    demand.gate_up_bias = gate_bias == invalid_tensor_handle ? nullptr : &model.weights.at(gate_bias);
    demand.down = borrow_resident_tensor(model.weights.at(ffn.down_weight));
    demand.down_bias = ffn.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(ffn.down_bias);
    demand.residency_group = ffn.residency_group;
    demand.activation_limit = ffn.activation_limit;
    demand.activation = ffn.activation;
    if (ffn.gate_up_weight == invalid_tensor_handle)
    {
        demand.up = borrow_resident_tensor(model.weights.at(ffn.up_weight));
        demand.up_bias = ffn.up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(ffn.up_bias);
    }
    if (!has_flag(model.opt.optimization_flags, OptimizationVulkanExpertGpuPriority))
    {
        if (input.rows() < vulkan_expert_gpu_admission_min_rows)
            return false;
        model.expert_backend->admit(demand);
        return false;
    }
    auto upload = model.expert_backend->begin_demand_batch({&demand, 1});
    if (!upload)
        return false;
    std::array<std::shared_ptr<const void>, 1> pins;
    if (upload->wait(pins) != 1 || !pins.front())
    {
        upload->abort();
        return false;
    }
    upload.reset();
    // Keep the demand pin until the execution submission has captured and
    // completed its entry. Both use the same eviction and publication rules.
    return execute() == ExpertStatus::Executed;
}

void forward_dense_ffn(const CompiledModel& model,
                       const ExpertPlan& ffn,
                       const ActivationBuffer& input,
                       ActivationBuffer& output,
                       ExpertWorkspace& workspace,
                       ExecutionBackend backend)
{
    if (ffn.use_gpu_cache)
    {
        if (backend == ExecutionBackend::Vulkan
            && forward_cached_dense_ffn(model, ffn, input, output))
            return;
        forward_expert(model, ffn, nullptr, input, output, workspace, false);
        return;
    }
    if (backend == ExecutionBackend::Vulkan)
    {
        const CompiledOperator& fused = model.operators.at(ffn.vulkan_operator);
        if (fused.qnk_expert)
        {
            if (fused.qnk_expert->forward(input, output))
                return;
            throw std::runtime_error("Vulkan dense FFN execution failed");
        }
        const CompiledOperator& down = model.operators.at_weight(ffn.down_weight);
        if (fused.bfloat16 && down.bfloat16)
        {
            if (fused.bfloat16->forward_swiglu_chain(input, *down.bfloat16,
                                                     model.weights.at(ffn.up_weight).shape[0],
                                                     ffn.activation, ffn.activation_limit, false, output))
                return;
            throw std::runtime_error("Vulkan dense FFN execution failed");
        }
        if (ffn.gate_up_weight == invalid_tensor_handle)
        {
            const uint64_t flags = model.opt.optimization_flags;
            forward_linear(model.weights.at(ffn.gate_weight), input, workspace.gate,
                           flags, model.operators.find_weight(ffn.gate_weight), backend, &workspace.quantized_input);
            forward_linear(model.weights.at(ffn.up_weight), input, workspace.projection,
                           flags, model.operators.find_weight(ffn.up_weight), backend, &workspace.quantized_input);
            for (size_t row = 0; row < input.rows(); ++row)
            {
                float* up = workspace.projection.row(row);
                const float* gate = workspace.gate.row(row);
                for (uint32_t column = 0; column < workspace.projection.columns(); ++column)
                    up[column] = gated_product(gate[column], up[column], ffn.activation,
                                               ffn.activation_limit, flags);
            }
            forward_linear(model.weights.at(ffn.down_weight), workspace.projection, output,
                           flags, &down, backend, &workspace.quantized_input);
            return;
        }
    }
    forward_expert(model, ffn, nullptr, input, output, workspace, false);
}

void forward_shared_expert(const CompiledModel& model,
                           const MoeBlockPlan& moe,
                           const ActivationBuffer& input,
                           ActivationBuffer& output,
                           ExpertWorkspace& workspace,
                           const std::shared_ptr<const DeviceTensor_vulkan>& device_input,
                           std::shared_ptr<const DeviceTensor_vulkan>* device_output,
                           bool require_device_execution)
{
    assert(&input != &output);
    if (device_output)
        device_output->reset();
    const ExpertPlan& expert = moe.shared_expert;
    const uint64_t optimization_flags = model.opt.optimization_flags;
    const bool has_router_gate = moe.shared_expert_gate_weight != invalid_tensor_handle;
    const auto& qnk_shared = model.operators.at(expert.vulkan_operator).qnk_expert;
    if (require_device_execution && (!device_output || !device_input || device_input->empty()))
        throw std::runtime_error("Vulkan Shared Expert requires a completed device input and output");
    if (qnk_shared && (!has_router_gate || qnk_shared->has_router_gate()))
    {
        if (device_output && device_input && !device_input->empty())
        {
            auto completed = std::make_shared<DeviceTensor_vulkan>();
            if (qnk_shared->forward_device(*device_input, *completed))
            {
                output.clear();
                *device_output = std::move(completed);
                return;
            }
            throw std::runtime_error("Vulkan native Shared Expert device execution failed");
        }
        if (!require_device_execution)
        {
            if (qnk_shared->forward(input, output))
                return;
            throw std::runtime_error("Vulkan native Shared Expert execution failed");
        }
    }
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
        if (!require_device_execution && float8_gate->forward_swiglu_chain(input, *float8_up, *float8_down, expert.activation, expert.activation_limit, output))
            return;
    }
    const CompiledOperator& fused_shared_operator = model.operators.at(moe.fused_shared_input_bfloat16_operator);
    bool shared_bfloat_chain_attempted = false;
    if (fused_shared_operator.bfloat16
        && expert.down_weight != invalid_tensor_handle)
    {
        const CompiledOperator& down_operator = model.operators.at_weight(expert.down_weight);
        if (down_operator.bfloat16)
        {
            shared_bfloat_chain_attempted = true;
            const uint32_t intermediate = model.weights.at(expert.up_weight).shape[0];
            if (device_output && device_input && !device_input->empty())
            {
                auto completed = std::make_shared<DeviceTensor_vulkan>();
                if (fused_shared_operator.bfloat16->forward_swiglu_chain_device(*device_input,
                                                                                *down_operator.bfloat16,
                                                                                intermediate,
                                                                                expert.activation,
                                                                                expert.activation_limit,
                                                                                has_router_gate,
                                                                                *completed))
                {
                    output.clear();
                    *device_output = std::move(completed);
                    return;
                }
            }
            if (!require_device_execution && fused_shared_operator.bfloat16->forward_swiglu_chain(input, *down_operator.bfloat16, intermediate, expert.activation, expert.activation_limit, has_router_gate, output))
            {
                return;
            }
        }
    }
    if (!require_device_execution && fused_shared_operator.bfloat16)
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
                        destination[column] = gated_product(source[column],
                                                            source[intermediate + column],
                                                            expert.activation,
                                                            expert.activation_limit,
                                                            optimization_flags);
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

    if (require_device_execution)
        throw std::runtime_error("Vulkan Shared Expert has no executable device chain or router gate");
    const ExecutionBackend backend = !shared_bfloat_chain_attempted
                                             && support_vulkan_shared_experts(model.operators, moe)
                                         ? ExecutionBackend::Vulkan
                                         : ExecutionBackend::Cpu;
    forward_dense_ffn(model, moe.shared_expert, input, output, workspace,
                      backend);
    if (moe.shared_expert_gate_weight == invalid_tensor_handle)
        return;
    ActivationBuffer& gate = workspace.gate;
    forward_linear(model.weights.at(moe.shared_expert_gate_weight),
                   input,
                   gate,
                   optimization_flags,
                   model.operators.find_weight(moe.shared_expert_gate_weight),
                   backend,
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

static void forward_experts(const CompiledModel& model,
                            const MoeBlockPlan& moe,
                            LayerState& layer_state,
                            std::span<const size_t> active_indices,
                            ExpertScratch& scratch,
                            bool prefetch)
{
    if (active_indices.empty())
        return;

    const bool direct_input = layer_state.normalized.rows() == 1
                              && layer_state.normalized.dtype() == DType::Float32
                              && !layer_state.normalized_device
                              && std::all_of(active_indices.begin(), active_indices.end(), [&](size_t active_index) {
                                     const auto& routes = layer_state.active_experts()[active_index].batch.routes;
                                     return routes.size() == 1 && routes.front().token_index == 0;
                                 });
    // Device routes defer Host gathering, while M1 CPU routes read normalized directly.
    for (size_t active_index : active_indices)
    {
        ExpertState& active = layer_state.active_experts()[active_index];
        if (direct_input)
        {
            active.input.clear();
            continue;
        }
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
            const ActivationBuffer& input = direct_input ? layer_state.normalized : active.input;
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
            task.input = &input;
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
    bool parallelize_matrices = false;
    expert_team_size = std::min(static_cast<int>(active_indices.size()), static_cast<int>(cpu_linear_num_threads()));
    parallelize_experts = expert_team_size > 1;
    if (parallelize_experts
        && expert_team_size <= static_cast<int>(cpu_linear_num_threads() / 2))
    {
        // BF16 Experts may underfill the outer team. Prefer matrix teams
        // only when every projection can use at least twice as many threads.
        parallelize_matrices = true;
        for (size_t active_index : active_indices)
        {
            const ExpertState& active = layer_state.active_experts()[active_index];
            const ActivationBuffer& input = direct_input ? layer_state.normalized : active.input;
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
                                                 * matrix->shape[1] * input.rows();
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
#if defined(_OPENMP) && (defined(__aarch64__) || defined(_M_ARM64))
    if (parallelize_matrices && active_indices.size() >= 2 && !prefetch
        && layer_state.normalized.rows() == 1
        && layer_state.normalized.dtype() == DType::Float32)
    {
        const size_t active_count = active_indices.size();
        const uint32_t input_columns = layer_state.normalized.columns();
        const auto executable = [](const CompiledOperator* op) {
            return op && (op->linear || op->bfloat16 || op->float8 || op->qnk || op->attention || op->latent_attention || op->latent_layer || op->layer_head || op->gated_delta);
        };
        const TensorData* first_gate_up = nullptr;
        const TensorData* first_down = nullptr;
        bool grouped_bfloat16 = input_columns != 0;
        for (size_t active_index : active_indices)
        {
            const ExpertState& active = layer_state.active_experts()[active_index];
            const ActivationBuffer& input = direct_input ? layer_state.normalized : active.input;
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            if (expert.layout != ExpertLayout::PackedGateUpDown
                || expert.gate_up_weight == invalid_tensor_handle
                || expert.down_weight == invalid_tensor_handle
                || active.batch.routes.size() != 1 || input.rows() != 1
                || input.columns() != input_columns
                || input.dtype() != DType::Float32
                || active.output.dtype() != DType::Float32
                || input.bytes().data() == active.output.bytes().data())
            {
                grouped_bfloat16 = false;
                break;
            }

            const TensorData& gate_up = active.lease.gate_up
                                            ? *active.lease.gate_up
                                            : model.weights.at(expert.gate_up_weight);
            const TensorData& down = active.lease.down
                                         ? *active.lease.down
                                         : model.weights.at(expert.down_weight);
            // The matrix-team check above already requires 2D BF16 weights.
            const CompiledOperator* gate_op = active.lease.gate_up
                                                  ? active.lease.gate_up_operator
                                                  : model.operators.find_weight(expert.gate_up_weight);
            const CompiledOperator* down_op = active.lease.down
                                                  ? active.lease.down_operator
                                                  : model.operators.find_weight(expert.down_weight);
            if (expert.gate_up_bias != invalid_tensor_handle
                || expert.down_bias != invalid_tensor_handle
                || gate_up.shape[0] % 2 != 0 || gate_up.shape[1] != input_columns
                || down.shape[0] != input_columns
                || down.shape[1] != gate_up.shape[0] / 2
                || gate_up.bfloat16_values().size() != gate_up.element_count()
                || down.bfloat16_values().size() != down.element_count()
                || executable(gate_op) || executable(down_op))
            {
                grouped_bfloat16 = false;
                break;
            }
            if (!first_gate_up)
            {
                first_gate_up = &gate_up;
                first_down = &down;
            }
            else if (gate_up.shape != first_gate_up->shape
                     || down.shape != first_down->shape)
            {
                grouped_bfloat16 = false;
                break;
            }
        }

        if (grouped_bfloat16)
        {
            const uint32_t intermediate_size = first_gate_up->shape[0] / 2;
            const uint32_t gate_rows = first_gate_up->shape[0];
            const uint32_t gate_groups = gate_rows / 8 + (gate_rows % 8 != 0);
            const uint32_t down_groups = input_columns / 8 + (input_columns % 8 != 0);
            const int64_t gate_tasks = static_cast<int64_t>(active_count * gate_groups);
            const int64_t down_tasks = static_cast<int64_t>(active_count * down_groups);
            const size_t gate_limit = input_columns <= 1024
                                          ? 4u * 1024u * 1024u
                                          : 512u * 1024u;
            const size_t down_limit = intermediate_size <= 1024
                                          ? 4u * 1024u * 1024u
                                          : 512u * 1024u;
            const bool gate_gemm = gate_rows >= 2 && input_columns >= 16
                                   && first_gate_up->element_count() <= gate_limit;
            const bool down_gemm = input_columns >= 2 && intermediate_size >= 16
                                   && first_down->element_count() <= down_limit;
            if (scratch.expert_workspaces.size() < active_count)
                scratch.expert_workspaces.resize(active_count);
            for (size_t slot = 0; slot < active_count; ++slot)
            {
                ExpertState& active = layer_state.active_experts()[active_indices[slot]];
                // Retain the gate/up stride across decode calls; down reads its first half.
                scratch.expert_workspaces[slot].projection.reset(1, gate_rows, false);
                active.output.reset(1, input_columns, false);
            }

#if defined(__GNUC__) || defined(__clang__)
            const auto forward = [&](size_t slot, uint32_t group, bool gate) __attribute__((always_inline)) {
#else
            const auto forward = [&](size_t slot, uint32_t group, bool gate) {
#endif
                ExpertState& active = layer_state.active_experts()[active_indices[slot]];
                const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                const TensorData& matrix = gate
                                               ? (active.lease.gate_up
                                                      ? *active.lease.gate_up
                                                      : model.weights.at(expert.gate_up_weight))
                                               : (active.lease.down
                                                      ? *active.lease.down
                                                      : model.weights.at(expert.down_weight));
                const uint32_t columns = gate ? input_columns : intermediate_size;
                const uint32_t output_columns = gate ? gate_rows : input_columns;
                const uint32_t first_row = group * 8;
                const uint32_t rows = std::min(8u, output_columns - first_row);
                const bool use_gemm = gate ? gate_gemm : down_gemm;
                const ActivationBuffer& source = direct_input ? layer_state.normalized : active.input;
                const float* input = gate
                                         ? source.row(0)
                                         : scratch.expert_workspaces[slot].projection.row(0);
                float* output = gate
                                    ? scratch.expert_workspaces[slot].projection.row(0)
                                    : active.output.row(0);
                const std::span<const uint16_t> weights = matrix.bfloat16_values();
                const uint16_t* weight = weights.data() + static_cast<size_t>(first_row) * columns;
                if (use_gemm)
                    bfloat16_gemm_4x8(weight, columns, input, columns, columns,
                                      rows, 1, output + first_row, output_columns);
                else
                    for (uint32_t row = 0; row < rows; ++row)
                        output[first_row + row] = bfloat16_dot(weights.data() + static_cast<size_t>(first_row + row) * columns,
                                                               input, columns);
            };

#pragma omp parallel num_threads(cpu_linear_num_threads())
            {
#pragma omp for schedule(static)
                for (int64_t task = 0; task < gate_tasks; ++task)
                    forward(static_cast<size_t>(task) / gate_groups,
                            static_cast<uint32_t>(task % gate_groups), true);

#pragma omp for schedule(static)
                for (int64_t task = 0; task < static_cast<int64_t>(active_count); ++task)
                {
                    const size_t slot = static_cast<size_t>(task);
                    ExpertState& active = layer_state.active_experts()[active_indices[slot]];
                    const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                    float* values = scratch.expert_workspaces[slot].projection.row(0);
                    const bool vector_silu = intermediate_size >= 4
                                             && expert.activation_limit <= 0.0f
                                             && has_flag(model.opt.optimization_flags,
                                                         OptimizationCpuFastSilu)
                                             && (expert.activation == ExpertActivation::Silu
                                                 || expert.activation == ExpertActivation::DeepSeekSwiGlu);
                    if (vector_silu)
                        float_silu_mul(values, values, values + intermediate_size,
                                       1.0f, 0.0f, intermediate_size);
                    else
                        for (uint32_t column = 0; column < intermediate_size; ++column)
                            values[column] = activate(values[column], expert.activation,
                                                      expert.activation_limit,
                                                      model.opt.optimization_flags)
                                             * values[intermediate_size + column];
                }

#pragma omp for schedule(static)
                for (int64_t task = 0; task < down_tasks; ++task)
                    forward(static_cast<size_t>(task) / down_groups,
                            static_cast<uint32_t>(task % down_groups), false);
            }
            return;
        }
    }
#endif
    const int64_t count = static_cast<int64_t>(active_indices.size());
    Bf16Counter* const bf16_counter = current_bf16_counter();
    const size_t workspace_count = parallelize_experts
                                       ? static_cast<size_t>(expert_team_size)
                                       : 1;
    if (scratch.expert_workspaces.size() < workspace_count)
        scratch.expert_workspaces.resize(workspace_count);
    const auto forward = [&](int64_t task_index, size_t workspace_index) {
        ExpertState& active = layer_state.active_experts()[active_indices[static_cast<size_t>(task_index)]];
        const uint32_t expert_id = active.batch.expert_id;
        const ActivationBuffer& input = direct_input ? layer_state.normalized : active.input;
        forward_expert(model,
                       moe.experts[expert_id],
                       active.lease.gate_up ? &active.lease : nullptr,
                       input,
                       active.output,
                       scratch.expert_workspaces[workspace_index],
                       prefetch && !prefetched_batch_weights);
    };
    if (!parallelize_experts)
    {
        const Bf16Scope bf16_scope(bf16_counter);
        for (int64_t task_index = 0; task_index < count; ++task_index)
            forward(task_index, 0);
        return;
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(expert_team_size)
    for (int64_t task_index = 0; task_index < count; ++task_index)
    {
        const Bf16Scope bf16_scope(bf16_counter);
        size_t workspace_index = 0;
#if defined(_OPENMP)
        workspace_index = static_cast<size_t>(omp_get_thread_num());
#endif
        forward(task_index, workspace_index);
    }
    return;
}

bool support_vulkan_expert(const ExpertPlan& expert,
                           const TensorData& gate_up,
                           const TensorData& down,
                           uint64_t optimization_flags,
                           const TensorData* up)
{
    if ((expert.activation != ExpertActivation::Silu
         && expert.activation != ExpertActivation::GptOssSwiGlu
         && expert.activation != ExpertActivation::DeepSeekSwiGlu)
        || gate_up.shape.size() != 2
        || down.shape.size() != 2
        || gate_up.shape[0] == 0 || gate_up.shape[1] == 0
        || down.shape[0] == 0 || down.shape[1] == 0
        || (up ? up->shape != gate_up.shape || down.shape[1] != gate_up.shape[0]
               : gate_up.shape[0] % 2 != 0 || down.shape[1] != gate_up.shape[0] / 2))
    {
        return false;
    }
    if (!up && gate_up.dtype == DType::MxFp4 && down.dtype == DType::MxFp4)
    {
        if (gate_up.input_column_indices.empty() && down.input_column_indices.empty())
            return true;
        return has_flag(optimization_flags, OptimizationVulkanQnK);
    }
    if (!up && gate_up.dtype == DType::BFloat16 && down.dtype == DType::BFloat16)
    {
        return expert.activation == ExpertActivation::Silu
               && gate_up.shape[0] / 2 % 128 == 0
               && gate_up.bfloat16_values().size() == gate_up.element_count()
               && down.bfloat16_values().size() == down.element_count();
    }
    const auto supported = [optimization_flags](const TensorData& weight) {
        if (weight.dtype == DType::BFloat16 || weight.dtype == DType::Float32)
            return true;
        if (weight.dtype == DType::Int4)
            return weight.shape.size() == 2 && weight.quantization_group_size != 0;
        return has_flag(optimization_flags, OptimizationVulkanQnK)
               && is_qnk_dtype(weight.dtype)
               && qnk_shape_supported(weight.dtype, weight.shape[0], weight.shape[1]);
    };
    const auto quantized = [](DType dtype) { return dtype == DType::Int4 || is_qnk_dtype(dtype); };
    const bool separate_floating = up
                                   && (gate_up.dtype == DType::BFloat16 || gate_up.dtype == DType::Float32)
                                   && (up->dtype == DType::BFloat16 || up->dtype == DType::Float32)
                                   && (down.dtype == DType::BFloat16 || down.dtype == DType::Float32);
    return (separate_floating || gate_up.dtype == DType::Float32 || down.dtype == DType::Float32
            || quantized(gate_up.dtype) || quantized(down.dtype) || (up && quantized(up->dtype)))
           && supported(gate_up) && supported(down) && (!up || supported(*up));
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
    ExpertDemandRequest demand;
    demand.key = expert.cache_key;
    demand.gate_up = lease.gate_up;
    demand.gate_up_bias = expert.gate_up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_bias);
    demand.down = lease.down;
    demand.down_bias = expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias);
    demand.residency_group = residency_group;
    demand.activation_limit = expert.activation_limit;
    demand.activation = expert.activation;
    model.expert_backend->admit(demand);
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
                         SessionStatistics* stats,
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
        LayerState& state;
        bool completed = false;
        ~WaveCleanup()
        {
            if (!completed)
                for (ExpertState& active : state.active_experts())
                    active.lease = {};
            scratch.backend_requests.clear();
            scratch.cache_leases.clear();
            scratch.prestaged_demands.clear();
            scratch.prestaged_pins.clear();
            scratch.prestaged_cache_leases.clear();
            scratch.current_wave_requests.clear();
            scratch.next_wave_requests.clear();
            scratch.wave_demands.clear();
            scratch.prepared_pins.clear();
        }
    } wave_cleanup{scratch, layer_state};
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
    const size_t active_expert_count = layer_state.active_experts().size();
    const bool direct_input = active_expert_count != 0
                              && layer_state.normalized.rows() == 1
                              && layer_state.normalized.dtype() == DType::Float32
                              && !layer_state.normalized_device
                              && std::all_of(layer_state.active_experts().begin(),
                                             layer_state.active_experts().end(),
                                             [](const ExpertState& active) {
                                                 return active.batch.routes.size() == 1
                                                        && active.batch.routes.front().token_index == 0;
                                             });
    const bool gpu_expert_batch_eligible = layer_state.normalized.rows() >= vulkan_expert_gpu_min_rows
                                           || model.opt.hybrid_mode == HybridMode::HybridExperts;
    const bool use_backend_batch = backend == ExecutionBackend::Vulkan
                                   && model.expert_backend
                                   && gpu_expert_batch_eligible
                                   && !has_flag(model.opt.flags, OptionDisableGpuExpertExecution);
    if (layer_state.require_device_execution
        && (!gpu_priority || !use_backend_batch || !layer_state.normalized_device || layer_state.normalized_device->empty()))
        return Error{ErrorCode::InternalError, "Vulkan routed Experts require GPU-priority execution and completed device input"};
    Result<void> device_execution_status;
    const auto forward_cpu = [&](std::span<const size_t> indices, const char* reason) {
        if (indices.empty()) return;
        if (layer_state.require_device_execution)
        {
            device_execution_status = Error{ErrorCode::InternalError, reason};
            return;
        }
        forward_experts(model, moe, layer_state, indices, scratch, prefetch);
        if (stats)
        {
            stats->expert_cpu_batches += indices.size();
            if (backend == ExecutionBackend::Vulkan)
                stats->expert_cpu_fallbacks += indices.size();
            for (size_t index : indices)
                stats->expert_cpu_routes += layer_state.active_experts()[index].batch.routes.size();
        }
    };
    const auto can_use_device_input = [&](const ExpertState& active) {
        if (!use_backend_batch || !layer_state.normalized_device || layer_state.normalized_device->empty()) return false;
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        const TensorHandle gate_handle = expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight : expert.gate_weight;
        if (gate_handle == invalid_tensor_handle || expert.down_weight == invalid_tensor_handle) return false;
        const TensorData& gate_up = model.weights.at(gate_handle);
        const TensorData& down = model.weights.at(expert.down_weight);
        const TensorData* up = expert.gate_up_weight == invalid_tensor_handle && expert.up_weight != invalid_tensor_handle
                                   ? &model.weights.at(expert.up_weight)
                                   : nullptr;
        return support_vulkan_expert(expert, gate_up, down, model.opt.optimization_flags, up)
               && ((gate_up.dtype == DType::MxFp4 && down.dtype == DType::MxFp4)
                   || (gate_up.dtype == DType::BFloat16 && down.dtype == DType::BFloat16)
                   || gate_up.dtype == DType::Float32 || down.dtype == DType::Float32
                   || gate_up.dtype == DType::Int4 || down.dtype == DType::Int4
                   || is_qnk_dtype(gate_up.dtype) || is_qnk_dtype(down.dtype));
    };
    if (layer_state.require_device_execution)
        for (const ExpertState& active : layer_state.active_experts())
            if (!can_use_device_input(active))
                return Error{ErrorCode::InternalError, "Vulkan routed Expert does not support device input for its native weight format or shape"};
#if defined(_OPENMP)
    int expert_team_size = 1;
    bool parallelize_regroup = false;
    expert_team_size = std::min(static_cast<int>(active_expert_count), static_cast<int>(cpu_linear_num_threads()));
    if (expert_team_size > 1 && !direct_input)
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
    if (direct_input)
    {
        for (ExpertState& active : layer_state.active_experts())
            active.input.clear();
    }
    else
    {
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
    }

    std::vector<size_t>& uncached = scratch.uncached_indices;
    std::vector<size_t>& pending = scratch.pending_indices;
    uncached.clear();
    pending.clear();
    uncached.reserve(active_expert_count);
    pending.reserve(active_expert_count);
    std::vector<ExpertStatus>& backend_status = scratch.backend_status;
    std::vector<uint8_t>& backend_aggregated = scratch.backend_aggregated;
    std::vector<size_t>& backend_indices = scratch.backend_indices;
    std::vector<ExpertRequest>& backend_requests = scratch.backend_requests;
    backend_status.clear();
    backend_aggregated.clear();
    backend_indices.clear();
    backend_requests.clear();
    scratch.backend_aggregated_output_valid = false;
    std::unique_ptr<ExpertSubmission> backend_submission;
    bool backend_reserved_work = false;
    bool backend_reservation_shape_valid = true;
    if (use_backend_batch)
    {
        backend_status.assign(active_expert_count, ExpertStatus::NotResident);
        backend_aggregated.assign(active_expert_count, 0);
        backend_indices.reserve(active_expert_count);
        backend_requests.reserve(active_expert_count);
        for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
        {
            ExpertState& active = layer_state.active_experts()[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            const TensorHandle gate_handle = expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight : expert.gate_weight;
            const TensorData* up_weight = expert.gate_up_weight == invalid_tensor_handle && expert.up_weight != invalid_tensor_handle
                                              ? &model.weights.at(expert.up_weight)
                                              : nullptr;
            if (gate_handle == invalid_tensor_handle
                || expert.down_weight == invalid_tensor_handle
                || !support_vulkan_expert(expert,
                                          model.weights.at(gate_handle),
                                          model.weights.at(expert.down_weight),
                                          model.opt.optimization_flags, up_weight))
            {
                continue;
            }
            const TensorData& gate_up_weight = model.weights.at(gate_handle);
            const TensorData& down_weight = model.weights.at(expert.down_weight);
            if (expert.cache_key.empty())
            {
                continue;
            }
            ExpertDemandRequest demand;
            demand.key = expert.cache_key;
            demand.gate_up = borrow_resident_tensor(gate_up_weight);
            demand.down = borrow_resident_tensor(down_weight);
            demand.down_bias = expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias);
            demand.residency_group = residency_group;
            demand.activation_limit = expert.activation_limit;
            demand.activation = expert.activation;
            bool can_admit = false;
            if (up_weight && !gate_up_weight.mxfp4_file_storage && !up_weight->mxfp4_file_storage && !down_weight.mxfp4_file_storage)
            {
                demand.gate_up_bias = expert.gate_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_bias);
                demand.up = borrow_resident_tensor(*up_weight);
                demand.up_bias = expert.up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.up_bias);
                can_admit = true;
            }
            else if (!up_weight && !gate_up_weight.mxfp4_file_storage && !down_weight.mxfp4_file_storage
                     && (gate_up_weight.dtype == DType::Int4 || is_qnk_dtype(gate_up_weight.dtype)
                         || down_weight.dtype == DType::Int4 || is_qnk_dtype(down_weight.dtype)))
            {
                demand.gate_up_bias = expert.gate_up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_bias);
                can_admit = true;
            }
            if (can_admit)
                model.expert_backend->admit(demand);
            backend_indices.push_back(active_index);
            const ActivationBuffer* input = nullptr;
            if (direct_input)
                input = &layer_state.normalized;
            else if (active.input.rows() != 0)
                input = &active.input;
            ExpertRequest request{expert.cache_key, input, &active.output, expert.weight_size, {}};
            request.input_rows = active.batch.routes.size();
            request.input_columns = layer_state.normalized.columns();
            request.route_aggregation.output = &scratch.backend_aggregated_output;
            request.route_aggregation.routes = active.batch.routes;
            request.route_aggregation.token_count = static_cast<uint32_t>(layer_state.normalized.rows());
            request.route_aggregation.completed = &backend_aggregated[active_index];
            request.device_input = layer_state.normalized_device;
            request.device_routes = active.batch.routes;
            if (can_use_device_input(active))
            {
                request.device_output = &active.device_output;
                request.route_aggregation = {};
            }
            backend_requests.push_back(request);
        }
        const bool complete_backend_coverage = backend_requests.size() == active_expert_count;
        if (complete_backend_coverage
            && std::any_of(backend_requests.begin(), backend_requests.end(), [](const ExpertRequest& request) {
                   return request.route_aggregation.output != nullptr;
               }))
            scratch.backend_aggregated_output.reset(layer_state.normalized.rows(), model.descriptor.hidden_size, true);
        for (ExpertRequest& request : backend_requests)
        {
            if (complete_backend_coverage)
                request.route_aggregation.require_all_requests = true;
            else
                request.route_aggregation = {};
        }
        backend_submission = model.expert_backend->submit_batch(backend_requests);
        if (backend_submission)
        {
            const std::span<const ExpertStatus> planned = backend_submission->reservations();
            backend_reservation_shape_valid = planned.size() == backend_indices.size();
            if (!backend_reservation_shape_valid)
            {
                // A malformed reservation is a failed backend response, not
                // a cold-cache miss eligible for demand upload.
                backend_submission->abort();
                for (size_t active_index : backend_indices)
                    backend_status[active_index] = ExpertStatus::Failed;
            }
            else
            {
                for (size_t result_index = 0; result_index < planned.size(); ++result_index)
                {
                    const size_t active_index = backend_indices[result_index];
                    if (planned[result_index] == ExpertStatus::Executed)
                    {
                        backend_status[active_index] = ExpertStatus::Executed;
                        backend_reserved_work = true;
                    }
                    else if (planned[result_index] != ExpertStatus::NotResident)
                    {
                        backend_status[active_index] = ExpertStatus::Failed;
                    }
                }
            }
        }
        else
        {
            for (size_t active_index : backend_indices)
                backend_status[active_index] = ExpertStatus::Failed;
        }
    }

    const auto finalize_backend = [&]() -> Result<void> {
        if (backend_submission)
        {
            std::vector<ExpertStatus>& backend_results = scratch.backend_results;
            backend_results.clear();
            backend_submission->wait(backend_results);
            bool backend_result_contract_valid = backend_reservation_shape_valid && backend_results.size() == backend_indices.size();
            if (backend_result_contract_valid)
            {
                const std::span<const ExpertStatus> planned = backend_submission->reservations();
                for (size_t result_index = 0; result_index < backend_results.size(); ++result_index)
                {
                    if (backend_results[result_index] == ExpertStatus::Executed
                        && planned[result_index] != ExpertStatus::Executed)
                    {
                        backend_result_contract_valid = false;
                        break;
                    }
                }
            }
            if (!backend_result_contract_valid)
            {
                for (size_t active_index : backend_indices)
                {
                    if (!gpu_priority && backend_status[active_index] == ExpertStatus::NotResident)
                        continue;
                    backend_status[active_index] = ExpertStatus::Failed;
                }
            }
            bool backend_has_executed = false;
            if (backend_result_contract_valid)
            {
                for (ExpertStatus result : backend_results)
                    backend_has_executed = backend_has_executed || result == ExpertStatus::Executed;
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
            for (size_t result_index = 0; result_index < backend_indices.size(); ++result_index)
            {
                const size_t active_index = backend_indices[result_index];
                // Non-priority misses have already completed on the CPU.
                // Only reserved work still needs recovery at this late wait.
                if (!gpu_priority
                    && backend_status[active_index] != ExpertStatus::Executed
                    && backend_status[active_index] != ExpertStatus::Failed)
                    continue;
                const ExpertStatus backend_result = result_index < backend_results.size()
                                                        ? backend_results[result_index]
                                                        : ExpertStatus::Failed;
                if (backend_result == ExpertStatus::Failed
                    || (backend_status[active_index] == ExpertStatus::Executed
                        && (!backend_commit_succeeded || backend_result != ExpertStatus::Executed)))
                {
                    backend_status[active_index] = ExpertStatus::Failed;
                    continue;
                }
            }
            backend_submission.reset();
        }

        std::vector<size_t>& recovery_indices = scratch.wave_failed_indices;
        recovery_indices.clear();
        recovery_indices.reserve(backend_indices.size());
        for (size_t active_index : backend_indices)
            if (backend_status[active_index] == ExpertStatus::Failed)
                recovery_indices.push_back(active_index);

        if (layer_state.require_device_execution && !recovery_indices.empty())
            return Error{ErrorCode::InternalError, "Vulkan resident Expert execution or publication failed"};
        if (!recovery_indices.empty())
        {
            size_t uncached_count = 0;
            for (size_t failed_index = 0; failed_index < recovery_indices.size(); ++failed_index)
            {
                const size_t active_index = recovery_indices[failed_index];
                ExpertState& active = layer_state.active_experts()[active_index];
                active.device_output.reset();
                const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                if (model.expert_cache && expert.gate_up_weight != invalid_tensor_handle
                    && !expert.cache_key.empty())
                {
                    const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                    const TensorData& down = model.weights.at(expert.down_weight);
                    const auto wait_start = expert_cache_wait_start(model);
                    const std::array<ExpertCachePairRequest, 1> requests{{{
                        &gate_up,
                        &down,
                        residency_group,
                        expert.cache_key,
                        victim_metadata(model, expert, layer_state.normalized.rows()),
                    }}};
                    std::array<ExpertCacheLease, 1> leases;
                    auto lease_count = model.expert_cache->wait_acquire_ready_pairs(requests, leases, true, 1, true, false);
                    if (!lease_count) return lease_count.error();
                    bool drained = false;
                    while (lease_count.value() == 0)
                    {
                        if (drained)
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        model.expert_backend->wait_for_background_work();
                        lease_count = model.expert_cache->wait_acquire_ready_pairs(requests, leases, true, 1, true, false);
                        if (!lease_count) return lease_count.error();
                        drained = true;
                    }
                    active.cache_wait_us += expert_cache_wait_elapsed(wait_start);
                    if (!leases[0].gate_up)
                        return Error{ErrorCode::InternalError, "Expert cache ready wait acquired no pair"};
                    active.lease = std::move(leases[0]);
                    forward_cpu(std::span<const size_t>(&active_index, 1), "Vulkan resident Expert execution or publication failed");
                    active.lease = {};
                }
                else
                {
                    recovery_indices[uncached_count++] = active_index;
                }
            }
            recovery_indices.resize(uncached_count);
            forward_cpu(recovery_indices, "Vulkan resident Expert execution or publication failed");
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
            if (backend_status[active_index] == ExpertStatus::NotResident
                && !expert.cache_key.empty())
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
    if (!prestaged_indices.empty()
        && std::any_of(prestaged_indices.begin(), prestaged_indices.end(), [&](size_t active_index) {
               return backend_status[active_index] == ExpertStatus::Failed;
           }))
    {
        // Keep completed uploads for healthy requests; failed requests must
        // not re-enter compute through the preloaded wave after CPU recovery.
        size_t count = 0;
        size_t prepared = 0;
        for (size_t index = 0; index < prestaged_indices.size(); ++index)
        {
            const size_t active_index = prestaged_indices[index];
            if (backend_status[active_index] == ExpertStatus::Failed)
            {
                layer_state.active_experts()[active_index].lease = {};
                if (index < prestaged_pins.size()) prestaged_pins[index].reset();
                continue;
            }
            prestaged_indices[count] = active_index;
            if (index < prestaged_prepared) ++prepared;
            if (count != index && index < prestaged_pins.size())
                prestaged_pins[count] = std::move(prestaged_pins[index]);
            ++count;
        }
        prestaged_indices.resize(count);
        if (!prestaged_pins.empty()) prestaged_pins.resize(count);
        prestaged_prepared = prepared;
    }

    for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
    {
        ExpertState& active = layer_state.active_experts()[active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        if (use_backend_batch && backend_status[active_index] != ExpertStatus::NotResident)
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
    std::vector<ExpertRequest>& current_wave_requests = scratch.current_wave_requests;
    current_wave_indices.clear();
    current_wave_requests.clear();
    std::unique_ptr<ExpertSubmission> current_wave_submission;
    const auto finish_current_wave = [&] {
        if (current_wave_requests.empty()) return;
        std::vector<ExpertStatus>& results = scratch.backend_results;
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
                    if (results[index] == ExpertStatus::Executed
                        && planned[index] != ExpertStatus::Executed) valid = false;
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
            if (valid && results[index] == ExpertStatus::Executed)
            {
                if (use_backend_batch)
                    backend_status[active_index] = ExpertStatus::Executed;
            }
            else
            {
                // This transient wave list owns demand-failure recovery;
                // leaving status NotResident prevents the late finalizer from recovering it twice.
                active.device_output.reset();
                failed_indices.push_back(active_index);
            }
        }
        // Release device Entry pins before a capacity-limited upload retries.
        current_wave_submission.reset();
        forward_cpu(failed_indices, "Vulkan demand Expert execution or publication failed");
        for (size_t active_index : current_wave_indices)
            layer_state.active_experts()[active_index].lease = {};
        current_wave_indices.clear();
        current_wave_requests.clear();
    };

    const auto forward_ready = [&](const std::vector<size_t>& indices,
                                   std::span<std::shared_ptr<const void>> preload_pins = {},
                                   size_t preload_prepared = 0) {
        if (!gpu_priority && !use_backend_batch)
        {
            for (size_t active_index : indices)
                layer_state.active_experts()[active_index].device_output.reset();
            forward_cpu(indices, "Vulkan Expert backend execution is unavailable");
            for (size_t active_index : indices)
                layer_state.active_experts()[active_index].lease = {};
            return;
        }

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
            const TensorHandle gate_handle = expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight : expert.gate_weight;
            if (gpu_priority && use_backend_batch && !expert.cache_key.empty()
                && gate_handle != invalid_tensor_handle && expert.down_weight != invalid_tensor_handle)
            {
                const auto gate_up = active.lease.gate_up ? active.lease.gate_up : borrow_resident_tensor(model.weights.at(gate_handle));
                const auto down = active.lease.down ? active.lease.down : borrow_resident_tensor(model.weights.at(expert.down_weight));
                const auto up = expert.gate_up_weight == invalid_tensor_handle && expert.up_weight != invalid_tensor_handle
                                    ? borrow_resident_tensor(model.weights.at(expert.up_weight))
                                    : nullptr;
                supported = support_vulkan_expert(expert, *gate_up, *down, model.opt.optimization_flags, up.get());
                if (supported)
                {
                    demands.push_back({expert.cache_key, gate_up,
                                       (up ? expert.gate_bias : expert.gate_up_bias) == invalid_tensor_handle
                                           ? nullptr
                                           : &model.weights.at(up ? expert.gate_bias : expert.gate_up_bias),
                                       down, expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias),
                                       residency_group, expert.activation_limit, expert.activation, up,
                                       expert.up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.up_bias)});
                    demand_indices.push_back(active_index);
                }
            }
            if (!supported)
            {
                active.device_output.reset();
                if (layer_state.require_device_execution)
                {
                    device_execution_status = Error{ErrorCode::InternalError, "Vulkan Expert weights, shape, cache key or placement is unsupported"};
                    return;
                }
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
                    if (!device_execution_status) return;
                    continue; // Retry after both current host/device pins release.
                }
                if (layer_state.require_device_execution)
                {
                    device_execution_status = Error{ErrorCode::InternalError, "Vulkan Expert demand upload could not reserve device cache capacity or prepare native weights"};
                    return;
                }
                const size_t active_index = demand_indices[cursor++];
                layer_state.active_experts()[active_index].device_output.reset();
                cpu_indices.push_back(active_index);
                continue;
            }
            std::vector<size_t>& next_indices = scratch.next_wave_indices;
            std::vector<ExpertRequest>& next_requests = scratch.next_wave_requests;
            next_indices.clear();
            next_requests.clear();
            next_indices.reserve(prepared);
            next_requests.reserve(prepared);
            for (size_t index = 0; index < prepared; ++index)
            {
                const size_t active_index = demand_indices[cursor + index];
                ExpertState& active = layer_state.active_experts()[active_index];
                const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                const ActivationBuffer* input = nullptr;
                if (direct_input)
                    input = &layer_state.normalized;
                else if (active.input.rows() != 0)
                    input = &active.input;
                ExpertRequest request{expert.cache_key, input, &active.output, expert.weight_size, {}};
                request.input_rows = active.batch.routes.size();
                request.input_columns = layer_state.normalized.columns();
                request.device_input = layer_state.normalized_device;
                request.device_routes = active.batch.routes;
                if (can_use_device_input(active))
                    request.device_output = &active.device_output;
                next_indices.push_back(active_index);
                next_requests.push_back(request);
            }
            // An upload-ready next wave cannot be lost while the preceding
            // wave commits: its bounded pins protect every selected pair.
            finish_current_wave();
            if (!device_execution_status) return;
            current_wave_indices.swap(next_indices);
            current_wave_requests.swap(next_requests);
            current_wave_submission = model.expert_backend->submit_batch(current_wave_requests);
            prepared_pins.clear(); // submit_batch owns the selected Entry pins.
            cursor += prepared;
        }
        forward_cpu(cpu_indices, "Vulkan Expert demand preparation or format is unsupported");
        for (size_t active_index : cpu_indices)
            layer_state.active_experts()[active_index].lease = {};
        // Keep bookkeeping capacity, but release the completed upload's Host
        // weights before waiting for the next host-cache batch. Only current
        // compute leases may remain pinned across that capacity-limited wait.
        demands.clear();
    };

    const auto preload_pins = std::span<std::shared_ptr<const void>>(prestaged_pins);
    forward_ready(prestaged_indices, preload_pins, prestaged_prepared);
    if (!device_execution_status) return device_execution_status.error();
    prestaged_pins.clear();
    prestaged_demands.clear();
    forward_ready(uncached);
    if (!device_execution_status) return device_execution_status.error();

    if (ready_batch_acquired)
    {
        forward_ready(pending);
        if (!device_execution_status) return device_execution_status.error();
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
        // Wait for admitted reads while current compute remains in flight. If
        // capacity blocked admission, release its pins before draining/retry.
        auto acquired = model.expert_cache->wait_acquire_ready_pairs(requests, leases, true, 1, true, false);
        if (!acquired) return acquired.error();
        bool drained = false;
        while (acquired.value() == 0)
        {
            if (!current_wave_requests.empty())
            {
                finish_current_wave();
                if (!device_execution_status) return device_execution_status.error();
                drained = false;
            }
            if (!model.expert_backend)
            {
                acquired = model.expert_cache->wait_acquire_ready_pairs(requests, leases, true);
                break;
            }
            if (drained)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            model.expert_backend->wait_for_background_work();
            acquired = model.expert_cache->wait_acquire_ready_pairs(requests, leases, true, 1, true, false);
            if (!acquired) return acquired.error();
            drained = true;
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
                active.cache_wait_us += cache_wait_microseconds;
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
        if (!device_execution_status) return device_execution_status.error();
    }

    finish_current_wave();
    if (!device_execution_status) return device_execution_status.error();

    if (!gpu_priority)
    {
        auto finalized = finalize_backend();
        if (!finalized) return finalized.error();
    }

    if (stats)
    {
        stats->expert_batches += active_expert_count;
        for (size_t index = 0; index < backend_status.size(); ++index)
            if (backend_status[index] == ExpertStatus::Executed)
                stats->expert_gpu_routes += layer_state.active_experts()[index].batch.routes.size();
        if (expert_cache_wait_timing_enabled(model))
        {
            for (const ExpertState& active : layer_state.active_experts())
            {
                for (const ExpertRoute& route : active.batch.routes)
                {
                    if (route.rank >= maximum_expert_route_ranks)
                        continue;
                    ++stats->expert_route_rank_demands[route.rank];
                    stats->expert_route_rank_demand_queue_time_microseconds[route.rank] += active.cache_wait_us;
                }
            }
        }
    }
    if (layer_state.require_device_execution)
        for (const ExpertState& active : layer_state.active_experts())
            if (!active.device_output || active.device_output->empty()
                || active.device_output->rows() != active.batch.routes.size())
                return Error{ErrorCode::InternalError, "Vulkan Expert completed without publishing its device output"};
    layer_state.experts_executed = true;
    wave_cleanup.completed = true;
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
