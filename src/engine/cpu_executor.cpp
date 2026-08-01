#include "cpu_executor.h"

#include "kernels/cpu_attention.h"
#include "kernels/cpu_batch.h"
#include "kernels/cpu_gated_delta_net.h"
#include "kernels/cpu_hyper_connection.h"
#include "kernels/cpu_latent_attention.h"
#include "kernels/cpu_ops.h"
#include "cpu_session_state.h"
#include "cpu_topology.h"
#include "expert_backend.h"
#include "storage/expert_cache.h"
#include "backends/ncnn/ncnn_attention.h"
#include "backends/ncnn/ncnn_linear.h"

#include "ncnn/moe/expert_dispatcher.h"
#include "ncnn/moe/runtime.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <string_view>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <immintrin.h>
#endif

namespace ncnn {
namespace moe {

static constexpr size_t expert_prefetch_limit_bytes = 4 * 1024;
static constexpr size_t assumed_cache_line_bytes = 64;

class OpenMpHybridTeamLimit
{
public:
    explicit OpenMpHybridTeamLimit(HybridMode mode) noexcept
    {
#if defined(_OPENMP)
        if (mode == HybridMode::CpuOnly)
            return;
        // Bound hybrid OpenMP work to physical cores.
        previous_ = omp_get_max_threads();
        static const uint32_t physical_cores =
            discover_cpu_topology().physical_core_count;
        if (physical_cores != 0 && previous_ > static_cast<int>(physical_cores))
        {
            omp_set_num_threads(static_cast<int>(physical_cores));
            changed_ = true;
        }
#else
        (void)mode;
#endif
    }

    ~OpenMpHybridTeamLimit()
    {
#if defined(_OPENMP)
        if (changed_)
            omp_set_num_threads(previous_);
#endif
    }

    OpenMpHybridTeamLimit(const OpenMpHybridTeamLimit&) = delete;
    OpenMpHybridTeamLimit& operator=(const OpenMpHybridTeamLimit&) = delete;

private:
#if defined(_OPENMP)
    int previous_ = 1;
    bool changed_ = false;
#endif
};

static void expand_hyper_connections(CpuBatch& hidden, uint32_t multiplier, CpuBatch& scratch)
{
    if (multiplier <= 1)
        return;
    scratch.reset(hidden.rows(), hidden.columns() * multiplier, false);
    for (size_t row_index = 0; row_index < hidden.rows(); ++row_index)
    {
        for (uint32_t copy = 0; copy < multiplier; ++copy)
            std::copy_n(hidden.row(row_index), hidden.columns(), scratch.row(row_index) + static_cast<size_t>(copy) * hidden.columns());
    }
    hidden.swap(scratch);
}

static uint64_t elapsed_microseconds(std::chrono::steady_clock::time_point start)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

static uint64_t saturating_weight_product(uint64_t bytes, size_t count) noexcept
{
    if (bytes != 0 && count > std::numeric_limits<uint64_t>::max() / bytes)
    {
        return std::numeric_limits<uint64_t>::max();
    }
    return bytes * static_cast<uint64_t>(count);
}

static void add_saturating(uint64_t value, uint64_t& destination) noexcept
{
    if (destination > std::numeric_limits<uint64_t>::max() - value)
    {
        destination = std::numeric_limits<uint64_t>::max();
    }
    else
    {
        destination += value;
    }
}

static void record_expert_weight_demand(const ExpertPlan& expert, size_t route_count, SessionStatistics& statistics) noexcept
{
    add_saturating(expert.weight_bytes, statistics.expert_batch_weight_bytes);
    add_saturating(saturating_weight_product(expert.weight_bytes, route_count), statistics.expert_route_weight_bytes);
}

struct VulkanExecutionSnapshot
{
    uint64_t dispatches = 0;
    uint64_t attention_blocks = 0;
    NcnnVulkanRuntimeCounters counters;
};

static VulkanExecutionSnapshot capture_vulkan_execution()
{
    VulkanExecutionSnapshot snapshot;
    snapshot.dispatches = NcnnLinearOperator::current_thread_vulkan_dispatches();
    snapshot.attention_blocks = NcnnVulkanAttentionOperator::current_thread_blocks();
    snapshot.counters = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
    return snapshot;
}

static VulkanExecutionSnapshot vulkan_execution_delta(const VulkanExecutionSnapshot& before)
{
    VulkanExecutionSnapshot delta;
    delta.dispatches = NcnnLinearOperator::current_thread_vulkan_dispatches() - before.dispatches;
    delta.attention_blocks = NcnnVulkanAttentionOperator::current_thread_blocks() - before.attention_blocks;
    const NcnnVulkanRuntimeCounters after = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
    delta.counters.compute_submissions = after.compute_submissions - before.counters.compute_submissions;
    delta.counters.submit_wait_time_microseconds =
        after.submit_wait_time_microseconds
        - before.counters.submit_wait_time_microseconds;
    delta.counters.batch_uploads = after.batch_uploads - before.counters.batch_uploads;
    delta.counters.batch_downloads = after.batch_downloads - before.counters.batch_downloads;
    delta.counters.auxiliary_uploads = after.auxiliary_uploads - before.counters.auxiliary_uploads;
    delta.counters.auxiliary_upload_bytes = after.auxiliary_upload_bytes - before.counters.auxiliary_upload_bytes;
    delta.counters.staging_slot_resizes = after.staging_slot_resizes - before.counters.staging_slot_resizes;
    delta.counters.staging_slot_reuses = after.staging_slot_reuses - before.counters.staging_slot_reuses;
    delta.counters.staging_slot_acquisitions = after.staging_slot_acquisitions - before.counters.staging_slot_acquisitions;
    delta.counters.staging_slot_contentions = after.staging_slot_contentions - before.counters.staging_slot_contentions;
    delta.counters.command_buffer_reuses = after.command_buffer_reuses - before.counters.command_buffer_reuses;
    delta.counters.attention_qkv_rope_fusions = after.attention_qkv_rope_fusions - before.counters.attention_qkv_rope_fusions;
    delta.counters.attention_qkv_ring_fusions = after.attention_qkv_ring_fusions - before.counters.attention_qkv_ring_fusions;
    delta.counters.attention_decode_sdpa_fusions = after.attention_decode_sdpa_fusions - before.counters.attention_decode_sdpa_fusions;
    delta.counters.kv_ring_appends = after.kv_ring_appends - before.counters.kv_ring_appends;
    delta.counters.kv_ring_resizes = after.kv_ring_resizes - before.counters.kv_ring_resizes;
    delta.counters.kv_ring_wrapped_views = after.kv_ring_wrapped_views - before.counters.kv_ring_wrapped_views;
    return delta;
}

static void record_captured_vulkan_execution_delta(SessionStatistics& statistics, const VulkanExecutionSnapshot& delta)
{
    statistics.vulkan_linear_dispatches += delta.dispatches;
    statistics.vulkan_attention_blocks += delta.attention_blocks;
    statistics.vulkan_compute_submissions += delta.counters.compute_submissions;
    statistics.vulkan_submit_wait_time_microseconds +=
        delta.counters.submit_wait_time_microseconds;
    statistics.vulkan_batch_uploads += delta.counters.batch_uploads;
    statistics.vulkan_batch_downloads += delta.counters.batch_downloads;
    statistics.vulkan_auxiliary_uploads += delta.counters.auxiliary_uploads;
    statistics.vulkan_auxiliary_upload_bytes += delta.counters.auxiliary_upload_bytes;
    statistics.vulkan_staging_slot_resizes += delta.counters.staging_slot_resizes;
    statistics.vulkan_staging_slot_reuses += delta.counters.staging_slot_reuses;
    statistics.vulkan_staging_slot_acquisitions += delta.counters.staging_slot_acquisitions;
    statistics.vulkan_staging_slot_contentions += delta.counters.staging_slot_contentions;
    statistics.vulkan_command_buffer_reuses += delta.counters.command_buffer_reuses;
    statistics.vulkan_attention_qkv_rope_fusions += delta.counters.attention_qkv_rope_fusions;
    statistics.vulkan_attention_qkv_ring_fusions += delta.counters.attention_qkv_ring_fusions;
    statistics.vulkan_attention_decode_sdpa_fusions += delta.counters.attention_decode_sdpa_fusions;
    statistics.vulkan_kv_ring_appends += delta.counters.kv_ring_appends;
    statistics.vulkan_kv_ring_resizes += delta.counters.kv_ring_resizes;
    statistics.vulkan_kv_ring_wrapped_views += delta.counters.kv_ring_wrapped_views;
}

static void record_vulkan_execution_delta(SessionStatistics& statistics, const VulkanExecutionSnapshot& before)
{
    record_captured_vulkan_execution_delta(statistics, vulkan_execution_delta(before));
}

static void record_expert_backend_delta(SessionStatistics& statistics, const ExpertBackendStatistics& before, const ExpertBackendStatistics& after)
{
    statistics.expert_gpu_cache_hits += after.hits - before.hits;
    statistics.expert_gpu_cache_misses += after.misses - before.misses;
    statistics.expert_gpu_cache_admissions += after.admissions - before.admissions;
    statistics.expert_gpu_cache_stores += after.stores - before.stores;
    statistics.expert_gpu_cache_evictions += after.evictions - before.evictions;
    statistics.expert_gpu_cache_dropped_admissions += after.dropped_admissions - before.dropped_admissions;
    statistics.expert_gpu_cache_bytes_uploaded += after.bytes_uploaded - before.bytes_uploaded;
    statistics.expert_gpu_cache_resident_bytes = after.resident_bytes;
    statistics.expert_gpu_cache_pending_bytes = after.pending_bytes;
    statistics.expert_gpu_executions += after.executions - before.executions;
    statistics.expert_gpu_execution_failures += after.execution_failures - before.execution_failures;
    statistics.expert_gpu_cpu_preferred += after.cpu_preferred - before.cpu_preferred;
    statistics.expert_gpu_execution_time_microseconds += after.execution_time_microseconds - before.execution_time_microseconds;
    statistics.expert_gpu_arc_recent_bytes = after.arc_recent_bytes;
    statistics.expert_gpu_arc_frequent_bytes = after.arc_frequent_bytes;
    statistics.expert_gpu_arc_recent_target_bytes = after.arc_recent_target_bytes;
    statistics.expert_gpu_arc_recent_ghost_bytes = after.arc_recent_ghost_bytes;
    statistics.expert_gpu_arc_frequent_ghost_bytes = after.arc_frequent_ghost_bytes;
    statistics.expert_gpu_device_source_hits += after.device_source_hits - before.device_source_hits;
    statistics.expert_gpu_device_source_misses += after.device_source_misses - before.device_source_misses;
    statistics.expert_gpu_device_source_executions += after.device_source_executions - before.device_source_executions;
    statistics.expert_gpu_device_source_execution_failures += after.device_source_execution_failures - before.device_source_execution_failures;
}

static void record_expert_victim_cache_delta(SessionStatistics& statistics, const ExpertVictimCacheStatistics& before, const ExpertVictimCacheStatistics& after)
{
    statistics.expert_gpu_victim_cache_hits += after.hits - before.hits;
    statistics.expert_gpu_victim_cache_misses += after.misses - before.misses;
    statistics.expert_gpu_victim_cache_admissions += after.admissions - before.admissions;
    statistics.expert_gpu_victim_cache_filtered_admissions += after.filtered_admissions - before.filtered_admissions;
    statistics.expert_gpu_victim_cache_reused_admissions += after.reused_admissions - before.reused_admissions;
    statistics.expert_gpu_victim_cache_probe_admissions += after.probe_admissions - before.probe_admissions;
    statistics.expert_gpu_victim_cache_stores += after.stores - before.stores;
    statistics.expert_gpu_victim_cache_evictions += after.evictions - before.evictions;
    statistics.expert_gpu_victim_cache_dropped_admissions += after.dropped_admissions - before.dropped_admissions;
    statistics.expert_gpu_victim_cache_restore_failures += after.restore_failures - before.restore_failures;
    statistics.expert_gpu_victim_cache_bytes_uploaded += after.bytes_uploaded - before.bytes_uploaded;
    statistics.expert_gpu_victim_cache_bytes_downloaded += after.bytes_downloaded - before.bytes_downloaded;
    statistics.expert_gpu_victim_cache_restore_time_microseconds += after.restore_time_microseconds - before.restore_time_microseconds;
    statistics.expert_gpu_victim_cache_mapped_stores += after.mapped_stores - before.mapped_stores;
    statistics.expert_gpu_victim_cache_mapped_restores += after.mapped_restores - before.mapped_restores;
    statistics.expert_gpu_victim_cache_resident_bytes = after.resident_bytes;
    statistics.expert_gpu_victim_cache_pending_bytes = after.pending_bytes;
}

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
    const size_t hinted_bytes = std::min(size, expert_prefetch_limit_bytes);
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t offset = 0; offset < hinted_bytes; offset += assumed_cache_line_bytes)
        prefetch_address(bytes + offset);
    return hinted_bytes;
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
    return 0;
}

static uint64_t prefetch_weight(const WeightTable& weights, TensorHandle handle)
{
    return handle == invalid_tensor_handle ? 0 : prefetch_tensor(weights.at(handle));
}

static float activate(float value, ExpertActivation activation, float limit)
{
    switch (activation)
    {
    case ExpertActivation::Relu: return std::max(0.0f, value);
    case ExpertActivation::Silu: return value / (1.0f + std::exp(-value));
    case ExpertActivation::Gelu: return 0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
    case ExpertActivation::ClampedSilu:
    {
        const float clamped = limit > 0.0f ? std::clamp(value, -limit, limit) : value;
        return clamped / (1.0f + std::exp(-clamped));
    }
    case ExpertActivation::DeepSeekSwiGlu:
    {
        const float clamped = limit > 0.0f ? std::min(value, limit) : value;
        return clamped / (1.0f + std::exp(-clamped));
    }
    case ExpertActivation::GptOssSwiGlu: return value;
    }
    return value;
}

static void record_mxfp4(const TensorData& matrix, size_t input_rows, ExpertExecutionMetrics& metrics)
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

static CpuBatch expert_linear(const TensorData& matrix, const TensorData* bias, const CpuBatch& input, ExpertExecutionMetrics& metrics)
{
    record_mxfp4(matrix, input.rows(), metrics);
    return bias ? linear_batch(matrix, *bias, input) : linear_batch(matrix, input);
}

static CpuBatch run_expert(const WeightTable& weights, const ExpertPlan& expert, const ExpertCacheLease* cached_weights, const CpuBatch& input, bool prefetch, ExpertExecutionMetrics& metrics)
{
    if (expert.gate_up_weight != invalid_tensor_handle)
    {
        const TensorData& gate_up_weight = cached_weights && cached_weights->gate_up ? *cached_weights->gate_up : weights.at(expert.gate_up_weight);
        if (prefetch)
            metrics.hinted_bytes += prefetch_tensor(gate_up_weight);
        const TensorData* gate_up_bias = expert.gate_up_bias == invalid_tensor_handle ? nullptr : &weights.at(expert.gate_up_bias);
        CpuBatch activated;
        if (gate_up_weight.dtype == DType::MxFp4)
        {
            activated = fused_mxfp4_gate_up_batch(gate_up_weight, gate_up_bias, input, expert.activation, expert.activation_limit);
            metrics.mxfp4_fused_gate_up_rows += static_cast<uint64_t>(activated.rows()) * activated.columns();
            record_mxfp4(gate_up_weight, input.rows(), metrics);
        }
        else if (has_flag(expert.flags, ExpertPlanPackedGateUp))
        {
            CpuBatch gate_up = expert_linear(gate_up_weight, gate_up_bias, input, metrics);
            activated = CpuBatch(gate_up.rows(), gate_up.columns() / 2);
            for (size_t token_index = 0; token_index < gate_up.rows(); ++token_index)
            {
                const float* source = gate_up.row(token_index);
                float* destination = activated.row(token_index);
                for (uint32_t column = 0; column < activated.columns(); ++column)
                {
                    const float gate = source[column];
                    const float up = source[activated.columns() + column];
                    destination[column] = activate(gate, expert.activation, expert.activation_limit) * up;
                }
            }
        }
        else
        {
            CpuBatch gate_up = expert_linear(gate_up_weight, gate_up_bias, input, metrics);
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
                    const float silu = gate / (1.0f + std::exp(-1.702f * gate));
                    destination[column] = silu * (linear + 1.0f);
                }
            }
        }
        const TensorData& down_weight = cached_weights && cached_weights->down ? *cached_weights->down : weights.at(expert.down_weight);
        if (prefetch)
            metrics.hinted_bytes += prefetch_tensor(down_weight);
        return expert_linear(down_weight, expert.down_bias == invalid_tensor_handle ? nullptr : &weights.at(expert.down_bias), activated, metrics);
    }

    if (prefetch)
        metrics.hinted_bytes += prefetch_weight(weights, expert.up_weight);
    const TensorData& up_weight = weights.at(expert.up_weight);
    if (has_flag(expert.flags, ExpertPlanGated) && expert.gate_weight != invalid_tensor_handle)
    {
        const TensorData& gate_weight = weights.at(expert.gate_weight);
        const TensorData& down_weight = weights.at(expert.down_weight);
        if (gate_weight.float8_linear_operator && up_weight.float8_linear_operator && down_weight.float8_linear_operator)
        {
            CpuBatch chained;
            if (gate_weight.float8_linear_operator->forward_swiglu_chain(
                    input,
                    *up_weight.float8_linear_operator,
                    *down_weight.float8_linear_operator,
                    expert.activation,
                    expert.activation_limit,
                    chained))
            {
                return chained;
            }
        }
    }
    CpuBatch up = expert_linear(up_weight, nullptr, input, metrics);
    if (has_flag(expert.flags, ExpertPlanGated))
    {
        if (prefetch)
            metrics.hinted_bytes += prefetch_weight(weights, expert.gate_weight);
        const CpuBatch gate = expert_linear(weights.at(expert.gate_weight), nullptr, input, metrics);
        for (size_t token_index = 0; token_index < up.rows(); ++token_index)
        {
            float* up_row = up.row(token_index);
            const float* gate_row = gate.row(token_index);
            for (uint32_t column = 0; column < up.columns(); ++column)
            {
                if (expert.activation == ExpertActivation::DeepSeekSwiGlu && expert.activation_limit > 0.0f)
                    up_row[column] = std::clamp(up_row[column], -expert.activation_limit, expert.activation_limit);
                up_row[column] *= activate(gate_row[column], expert.activation, expert.activation_limit);
            }
        }
    }
    else
    {
        for (size_t token_index = 0; token_index < up.rows(); ++token_index)
        {
            float* token = up.row(token_index);
            for (uint32_t column = 0; column < up.columns(); ++column)
                token[column] = activate(token[column], expert.activation, expert.activation_limit);
        }
    }
    if (prefetch)
        metrics.hinted_bytes += prefetch_weight(weights, expert.down_weight);
    return expert_linear(weights.at(expert.down_weight), nullptr, up, metrics);
}

static CpuBatch run_shared_expert(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    const CpuBatch& input,
    ExpertExecutionMetrics& metrics)
{
    if (moe.fused_shared_input_bfloat16_operator)
    {
        CpuBatch fused;
        if (moe.fused_shared_input_bfloat16_operator->forward(
                input,
                fused))
        {
            const ExpertPlan& expert = moe.shared_expert;
            const uint32_t intermediate =
                model.weights.at(expert.up_weight).shape[0];
            const bool has_router_gate =
                moe.shared_expert_gate_weight
                != invalid_tensor_handle;
            const uint32_t expected_columns =
                intermediate * 2 + (has_router_gate ? 1 : 0);
            if (fused.columns() == expected_columns)
            {
                CpuBatch activated(
                    fused.rows(),
                    intermediate);
                for (size_t token_index = 0;
                     token_index < fused.rows();
                     ++token_index)
                {
                    const float* source =
                        fused.row(token_index);
                    float* destination =
                        activated.row(token_index);
                    for (uint32_t column = 0;
                         column < intermediate;
                         ++column)
                    {
                        destination[column] =
                            activate(
                                source[column],
                                expert.activation,
                                expert.activation_limit)
                            * source[intermediate + column];
                    }
                }
                CpuBatch output = expert_linear(
                    model.weights.at(expert.down_weight),
                    nullptr,
                    activated,
                    metrics);
                if (has_router_gate)
                {
                    for (size_t token_index = 0;
                         token_index < output.rows();
                         ++token_index)
                    {
                        const float scale =
                            1.0f
                            / (1.0f
                               + std::exp(
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
        moe.shared_expert,
        nullptr,
        input,
        false,
        metrics);
    if (moe.shared_expert_gate_weight == invalid_tensor_handle)
        return output;
    CpuBatch gate = linear_batch(
        model.weights.at(moe.shared_expert_gate_weight),
        input);
    assert(gate.columns() == 1);
    for (size_t token_index = 0; token_index < output.rows(); ++token_index)
    {
        const float scale = 1.0f / (1.0f + std::exp(-gate.row(token_index)[0]));
        float* token = output.row(token_index);
        for (uint32_t column = 0; column < output.columns(); ++column)
            token[column] *= scale;
    }
    return output;
}

static bool can_overlap_vulkan_shared_expert(const CompiledModel& model, const MoeBlockPlan& moe)
{
    if (!moe.has_shared_expert)
        return false;
    const ExpertPlan& expert = moe.shared_expert;
    const auto uses_vulkan = [](const TensorData& tensor) {
        return tensor.bfloat16_linear_operator
               || tensor.float8_linear_operator
               || (tensor.linear_operator
                   && tensor.linear_operator->uses_vulkan());
    };
    if (moe.fused_shared_input_bfloat16_operator)
    {
        return expert.down_weight != invalid_tensor_handle
               && uses_vulkan(
                   model.weights.at(expert.down_weight));
    }
    if (expert.gate_up_weight != invalid_tensor_handle
        || expert.gate_weight == invalid_tensor_handle
        || expert.up_weight == invalid_tensor_handle
        || expert.down_weight == invalid_tensor_handle)
    {
        return false;
    }
    const TensorData& gate = model.weights.at(expert.gate_weight);
    const TensorData& up = model.weights.at(expert.up_weight);
    const TensorData& down = model.weights.at(expert.down_weight);
    if (!uses_vulkan(gate) || !uses_vulkan(up) || !uses_vulkan(down))
        return false;
    if (moe.shared_expert_gate_weight == invalid_tensor_handle)
        return true;
    return uses_vulkan(model.weights.at(moe.shared_expert_gate_weight));
}

static void capture_speculative_hidden(
    const CpuBatch& hidden,
    uint32_t hidden_size,
    uint32_t multiplier,
    size_t target_index,
    CpuBatch& destination)
{
    for (size_t row = 0; row < hidden.rows(); ++row)
    {
        float* output = destination.row(row) + target_index * hidden_size;
        const float* input = hidden.row(row);
        for (uint32_t copy = 0; copy < multiplier; ++copy)
        {
            const float* source = input + static_cast<size_t>(copy) * hidden_size;
            for (uint32_t column = 0; column < hidden_size; ++column)
                output[column] += source[column];
        }
        const float inverse_multiplier = 1.0f / static_cast<float>(multiplier);
        for (uint32_t column = 0; column < hidden_size; ++column)
            output[column] *= inverse_multiplier;
    }
}

struct OverlappedSharedExpertResult
{
    CpuBatch output;
    VulkanExecutionSnapshot vulkan;
};

static OverlappedSharedExpertResult run_overlapped_shared_expert(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    const CpuBatch& input)
{
    const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution();
    ExpertExecutionMetrics metrics;
    OverlappedSharedExpertResult result;
    result.output = run_shared_expert(model, moe, input, metrics);
    result.vulkan = vulkan_execution_delta(vulkan_before);
    return result;
}

static void gather_tokens(const CpuBatch& source, const std::vector<ExpertRoute>& routes, CpuBatch& gathered)
{
    gathered.reset(routes.size(), source.columns(), false);
    for (size_t route_index = 0; route_index < routes.size(); ++route_index)
    {
        std::copy_n(source.row(routes[route_index].token_index), source.columns(), gathered.row(route_index));
    }
}

static uint64_t run_experts(const CompiledModel& model, const MoeBlockPlan& moe, LayerGraphState& layer_state, std::span<const size_t> active_indices, CpuExpertExecutionScratch& scratch)
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
        const TensorData* gate_up = active.lease.gate_up
                                        ? active.lease.gate_up.get()
                                    : expert.gate_up_weight == invalid_tensor_handle
                                        ? nullptr
                                        : &model.weights.at(expert.gate_up_weight);
        const TensorData* down = active.lease.down
                                     ? active.lease.down.get()
                                 : expert.down_weight == invalid_tensor_handle
                                     ? nullptr
                                     : &model.weights.at(expert.down_weight);
        Mxfp4Task task;
        task.gate_up = gate_up;
        task.gate_up_bias = expert.gate_up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_bias);
        task.down = down;
        task.down_bias = expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias);
        task.input = &active.input;
        task.output = &active.output;
        task.activation = expert.activation;
        task.activation_limit = expert.activation_limit;
        decode_tasks.push_back(task);
    }

    const bool grouped_decode = mxfp4_expert_batch(decode_tasks, &scratch.kernels);
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
                model.expert_backend->observe_cpu(static_cast<uint32_t>(active.input.rows()), expert.weight_bytes, share);
            }
        }
        return elapsed;
    }

    bool parallelize_experts = false;
    int expert_team_size = 1;
#if defined(_OPENMP)
    expert_team_size = std::min(static_cast<int>(active_indices.size()), omp_get_max_threads());
    parallelize_experts = expert_team_size > 1;
#endif
    const int64_t parallel_expert_count = static_cast<int64_t>(active_indices.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(expert_team_size) if (parallelize_experts)
    for (int64_t task_index = 0; task_index < parallel_expert_count; ++task_index)
    {
        ActiveExpertExecution& active = layer_state.active_experts[active_indices[static_cast<size_t>(task_index)]];
        const uint32_t expert_id = active.batch.expert_id;
        active.output = run_expert(model.weights, moe.experts[expert_id], active.lease.gate_up ? &active.lease : nullptr, active.input, model.hybrid_mode == HybridMode::VulkanWithCpuPrefetch, active.metrics);
    }
    const uint64_t elapsed = elapsed_microseconds(compute_start);
    if (model.expert_backend)
    {
        const uint64_t share = std::max<uint64_t>(1, elapsed / active_indices.size());
        for (size_t active_index : active_indices)
        {
            const ActiveExpertExecution& active = layer_state.active_experts[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            model.expert_backend->observe_cpu(static_cast<uint32_t>(active.input.rows()), expert.weight_bytes, share);
        }
    }
    return elapsed;
}

static bool can_run_vulkan_expert(const ExpertPlan& expert, const TensorData& gate_up, const TensorData& down)
{
    return (expert.activation == ExpertActivation::GptOssSwiGlu || expert.activation == ExpertActivation::DeepSeekSwiGlu)
           && gate_up.dtype == DType::MxFp4
           && down.dtype == DType::MxFp4
           && gate_up.shape.size() == 2
           && down.shape.size() == 2
           && gate_up.shape[0] % 2 == 0
           && down.shape[1] == gate_up.shape[0] / 2;
}

static void admit_vulkan_expert(const CompiledModel& model, const ExpertPlan& expert, const ExpertCacheLease& lease, uint32_t residency_group, uint32_t token_count)
{
    if (!model.expert_backend || !lease.gate_up || !lease.down || !can_run_vulkan_expert(expert, *lease.gate_up, *lease.down))
    {
        return;
    }
    model.expert_backend->admit(expert.cache_key, lease.gate_up, expert.gate_up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_bias), lease.down,
                                expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias), residency_group, token_count,
                                expert.activation_limit, expert.activation);
}

static ExpertVictimExecutionMetadata victim_metadata(const CompiledModel& model, const ExpertPlan& expert)
{
    ExpertVictimExecutionMetadata metadata;
    if (expert.gate_up_weight == invalid_tensor_handle
        || expert.down_weight == invalid_tensor_handle
        || !can_run_vulkan_expert(expert, model.weights.at(expert.gate_up_weight), model.weights.at(expert.down_weight)))
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

static void collect_ranked_experts(
    const ExpertDispatchPlan& plan,
    std::span<uint32_t> ranked)
{
    const uint32_t invalid_expert = std::numeric_limits<uint32_t>::max();
    std::fill(ranked.begin(), ranked.end(), invalid_expert);
    for (const ExpertBatch& batch : plan.batches)
    {
        for (const ExpertRoute& route : batch.routes)
        {
            if (route.token_index == 0 && route.rank < ranked.size())
                ranked[route.rank] = batch.expert_id;
        }
    }
}

static void configure_router_prefetch(
    RouterPrefetchState& prefetch,
    uint32_t target_top_k)
{
    if (prefetch.target_top_k == target_top_k)
        return;
    prefetch = {};
    prefetch.target_top_k = target_top_k;
    prefetch.prefetch_width = std::min(2u, target_top_k);
}

static void adapt_router_prefetch_width(
    RouterPrefetchState& prefetch,
    const SessionStatistics& statistics,
    bool adaptive)
{
    if (!adaptive)
    {
        prefetch.prefetch_width = prefetch.target_top_k;
        return;
    }
    if (prefetch.prefetch_width == 0)
        prefetch.prefetch_width = std::min(2u, prefetch.target_top_k);
    ++prefetch.decisions;
    if (prefetch.decisions % 16 != 0)
        return;

    if (prefetch.prefetch_width > 1)
    {
        const uint32_t marginal_rank = prefetch.prefetch_width - 1;
        const uint64_t predictions = statistics.expert_route_rank_predictions[marginal_rank];
        const uint64_t matches = statistics.expert_route_rank_matches[marginal_rank];
        if (predictions >= 8 && matches * 3 < predictions)
        {
            --prefetch.prefetch_width;
            prefetch.last_adjustment_decision = prefetch.decisions;
            return;
        }
    }
    if (prefetch.decisions - prefetch.last_adjustment_decision < 64)
        return;
    if (prefetch.prefetch_width < prefetch.target_top_k)
    {
        const uint32_t widest_prefetched_rank = prefetch.prefetch_width - 1;
        const uint64_t predictions = statistics.expert_route_rank_predictions[widest_prefetched_rank];
        const uint64_t matches = statistics.expert_route_rank_matches[widest_prefetched_rank];
        const uint32_t next_rank = prefetch.prefetch_width;
        const uint64_t demands = statistics.expert_route_rank_demands[next_rank];
        const uint64_t queued_microseconds = statistics.expert_route_rank_demand_queue_time_microseconds[next_rank];
        if (predictions >= 8
            && matches * 2 >= predictions
            && demands >= 8
            && queued_microseconds / demands >= 2000)
        {
            ++prefetch.prefetch_width;
            prefetch.last_adjustment_decision = prefetch.decisions;
        }
    }
}

static void resolve_router_predictions(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    const ExpertDispatchPlan& plan,
    CpuSessionState& state,
    SessionStatistics& statistics,
    bool resolve_unused_predictions)
{
    if (!model.expert_cache
        || layer.layer_id >= state.layers.size()
        || !has_flag(model.runtime_option_flags, RuntimeOptionRouterPrediction))
    {
        return;
    }

    const uint32_t invalid_expert = std::numeric_limits<uint32_t>::max();
    CpuLayerCache& cache = state.layers[layer.layer_id];
    std::array<uint32_t, maximum_expert_route_ranks> actual_storage;
    std::vector<uint32_t> actual_fallback;
    std::span<const uint32_t> actual_experts;
    if (layer.moe.top_k <= actual_storage.size())
    {
        const std::span<uint32_t> ranked =
            std::span<uint32_t>(actual_storage).first(layer.moe.top_k);
        collect_ranked_experts(plan, ranked);
        actual_experts = ranked;
    }
    else
    {
        actual_fallback.resize(layer.moe.top_k);
        collect_ranked_experts(plan, actual_fallback);
        actual_experts = actual_fallback;
    }

    for (uint32_t rank = 0; rank < actual_experts.size() && rank < cache.predicted_expert_ids.size(); ++rank)
    {
        const uint32_t predicted = cache.predicted_expert_ids[rank];
        if (predicted == invalid_expert)
            continue;
        if (std::find(actual_experts.begin(), actual_experts.end(), predicted)
            != actual_experts.end())
        {
            ++statistics.expert_route_prediction_matches;
            ++statistics.expert_route_rank_matches[rank];
        }
    }

    std::array<std::string_view, maximum_expert_route_ranks> demanded_keys;
    size_t demanded_key_count = 0;
    for (uint32_t expert_id : actual_experts)
    {
        if (expert_id == invalid_expert
            || expert_id >= layer.moe.experts.size()
            || demanded_key_count == demanded_keys.size())
        {
            continue;
        }
        demanded_keys[demanded_key_count++] = layer.moe.experts[expert_id].cache_key;
    }
    if (resolve_unused_predictions && model.expected_concurrency == 1)
    {
        model.expert_cache->resolve_predictions(
            layer.layer_id,
            std::span<const std::string_view>(demanded_keys.data(), demanded_key_count));
    }

    cache.predicted_expert_ids.clear();
}

struct RouterPredictionTarget
{
    const CompiledLayerPlan* layer = nullptr;
    uint32_t prefetch_width = 0;
};

struct RouterPredictionOutcome
{
    uint32_t target_layer_id = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> predicted_expert_ids;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t predictor_time_microseconds = 0;
};

struct PendingRouterPrediction
{
    ~PendingRouterPrediction()
    {
        if (!result.valid())
            return;
        try
        {
            result.wait();
        }
        catch (...)
        {
        }
    }

    uint32_t target_layer_id = std::numeric_limits<uint32_t>::max();
    std::future<Result<RouterPredictionOutcome>> result;
};

static RouterPredictionTarget prepare_next_router_prediction(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    const CpuBatch& router_input,
    CpuSessionState& state,
    SessionStatistics& statistics)
{
    if (!model.expert_cache
        || model.expected_concurrency != 1
        || router_input.rows() != 1
        || layer.layer_id >= state.layers.size()
        || layer.moe.router_weight == invalid_tensor_handle
        || layer.moe.token_experts != invalid_tensor_handle
        || !has_flag(model.runtime_option_flags, RuntimeOptionRouterPrediction))
    {
        return {};
    }

    const CompiledLayerPlan* next_layer = nullptr;
    for (size_t layer_index = static_cast<size_t>(layer.layer_id) + 1;
         layer_index < model.layers.size();
         ++layer_index)
    {
        const CompiledLayerPlan& candidate = model.layers[layer_index];
        if (candidate.moe.router_weight != invalid_tensor_handle
            && candidate.moe.token_experts == invalid_tensor_handle
            && candidate.moe.top_k != 0
            && !candidate.moe.experts.empty())
        {
            next_layer = &candidate;
            break;
        }
    }
    if (!next_layer || next_layer->layer_id >= state.layers.size())
        return {};

    CpuLayerCache& source_cache = state.layers[layer.layer_id];
    configure_router_prefetch(
        source_cache.next_router_prediction,
        next_layer->moe.top_k);
    adapt_router_prefetch_width(
        source_cache.next_router_prediction,
        statistics,
        has_flag(model.runtime_option_flags, RuntimeOptionRankAdaptivePrefetch));
    return RouterPredictionTarget{
        next_layer,
        std::min(
            source_cache.next_router_prediction.prefetch_width,
            next_layer->moe.top_k)};
}

static Result<RouterPredictionOutcome> run_router_prediction(
    const CompiledModel& model,
    const CompiledLayerPlan& next_layer,
    const CpuBatch& router_input,
    uint32_t prefetch_width)
{
    const auto started = std::chrono::steady_clock::now();
    CpuBatch predicted_logits;
    linear_batch_into(
        model.weights.at(next_layer.moe.router_weight),
        router_input,
        predicted_logits);
    if (next_layer.moe.router_bias != invalid_tensor_handle)
    {
        add_bias_inplace(
            predicted_logits,
            model.weights.at(next_layer.moe.router_bias));
    }

    ExpertDispatchOptions options;
    options.expert_count = static_cast<uint32_t>(next_layer.moe.experts.size());
    options.top_k = next_layer.moe.top_k;
    options.score_function = next_layer.moe.score_function;
    options.normalization = next_layer.moe.normalization;
    options.routed_scaling_factor = next_layer.moe.routed_scaling_factor;
    if (next_layer.moe.router_selection_bias != invalid_tensor_handle)
    {
        options.selection_bias =
            model.weights.at(next_layer.moe.router_selection_bias).float32_values();
    }
    options.flags = has_flag(next_layer.moe.flags, MoeBlockNormalizeTopKWeights)
                        ? static_cast<uint32_t>(ExpertDispatchNormalizeTopKWeights)
                        : 0u;
    ExpertDispatcher dispatcher;
    ExpertDispatchPlan predicted_plan;
    auto dispatched = dispatcher.dispatch_into(
        predicted_logits.values(),
        1,
        options,
        predicted_plan);
    if (!dispatched)
        return dispatched.error();

    const uint32_t invalid_expert = std::numeric_limits<uint32_t>::max();
    std::array<uint32_t, maximum_expert_route_ranks> predicted_storage;
    std::vector<uint32_t> predicted_fallback;
    std::span<const uint32_t> predicted_experts;
    if (next_layer.moe.top_k <= predicted_storage.size())
    {
        const std::span<uint32_t> ranked =
            std::span<uint32_t>(predicted_storage).first(next_layer.moe.top_k);
        collect_ranked_experts(predicted_plan, ranked);
        predicted_experts = ranked;
    }
    else
    {
        predicted_fallback.resize(next_layer.moe.top_k);
        collect_ranked_experts(predicted_plan, predicted_fallback);
        predicted_experts = predicted_fallback;
    }
    RouterPredictionOutcome outcome;
    outcome.target_layer_id = next_layer.layer_id;
    outcome.predicted_expert_ids.assign(
        next_layer.moe.top_k,
        invalid_expert);
    const uint32_t width = std::min(prefetch_width, next_layer.moe.top_k);
    for (uint32_t rank = 0; rank < width; ++rank)
    {
        const uint32_t expert_id = predicted_experts[rank];
        if (expert_id == invalid_expert
            || expert_id >= next_layer.moe.experts.size())
            continue;
        const ExpertPlan& predicted = next_layer.moe.experts[expert_id];
        if (predicted.gate_up_weight == invalid_tensor_handle)
            continue;
        outcome.predicted_expert_ids[rank] = expert_id;
        const TensorData& gate_up = model.weights.at(predicted.gate_up_weight);
        const TensorData& down = model.weights.at(predicted.down_weight);
        if (!gate_up.mxfp4_file_storage && !down.mxfp4_file_storage)
            continue;
        const auto prediction = model.expert_cache->prefetch_pair(
            gate_up,
            down,
            next_layer.layer_id,
            predicted.cache_key);
        if (prediction && prediction.value())
            ++outcome.cache_hits;
        else
            ++outcome.cache_misses;
    }
    outcome.predictor_time_microseconds = elapsed_microseconds(started);
    return outcome;
}

static Result<void> apply_router_prediction(
    RouterPredictionOutcome outcome,
    CpuSessionState& state,
    SessionStatistics& statistics)
{
    if (outcome.target_layer_id >= state.layers.size())
    {
        return Error{
            ErrorCode::InternalError,
            "Router prediction target layer is outside Session state"};
    }
    const uint32_t invalid_expert = std::numeric_limits<uint32_t>::max();
    for (uint32_t rank = 0;
         rank < outcome.predicted_expert_ids.size()
         && rank < maximum_expert_route_ranks;
         ++rank)
    {
        if (outcome.predicted_expert_ids[rank] == invalid_expert)
            continue;
        ++statistics.expert_route_predictions;
        ++statistics.expert_route_rank_predictions[rank];
    }
    state.layers[outcome.target_layer_id].predicted_expert_ids =
        std::move(outcome.predicted_expert_ids);
    statistics.expert_route_prediction_cache_hits += outcome.cache_hits;
    statistics.expert_route_prediction_cache_misses += outcome.cache_misses;
    statistics.expert_route_prediction_time_microseconds +=
        outcome.predictor_time_microseconds;
    return {};
}

static Result<void> complete_router_prediction(
    uint32_t layer_id,
    PendingRouterPrediction& pending,
    CpuSessionState& state,
    SessionStatistics& statistics)
{
    if (!pending.result.valid())
        return {};
    if (pending.target_layer_id != layer_id)
    {
        if (pending.target_layer_id > layer_id)
            return {};
        return Error{
            ErrorCode::InternalError,
            "Router prediction completed after its target layer"};
    }

    const auto wait_started = std::chrono::steady_clock::now();
    Result<RouterPredictionOutcome> completed = Error{
        ErrorCode::InternalError,
        "Router prediction worker did not return a result"};
    try
    {
        completed = pending.result.get();
    }
    catch (const std::exception& error)
    {
        completed = Error{
            ErrorCode::InternalError,
            std::string("Router prediction worker failed: ") + error.what()};
    }
    catch (...)
    {
        completed = Error{
            ErrorCode::InternalError,
            "Router prediction worker failed"};
    }
    statistics.expert_route_prediction_wait_time_microseconds +=
        elapsed_microseconds(wait_started);
    pending.target_layer_id = std::numeric_limits<uint32_t>::max();
    if (!completed)
        return completed.error();
    ++statistics.expert_route_prediction_async_completions;
    return apply_router_prediction(
        std::move(completed).value(),
        state,
        statistics);
}

static Result<void> predict_next_router_routes(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    const CpuBatch& router_input,
    CpuSessionState& state,
    SessionStatistics& statistics,
    PendingRouterPrediction& pending)
{
    const RouterPredictionTarget target = prepare_next_router_prediction(
        model,
        layer,
        router_input,
        state,
        statistics);
    if (!target.layer)
        return {};

    if (has_flag(
            model.runtime_option_flags,
            RuntimeOptionAsyncRouterPrediction))
    {
        bool submitted = false;
        try
        {
            if (!state.router_prediction_worker)
            {
                state.router_prediction_worker =
                    std::make_unique<CpuTaskWorker>(2);
            }
            CpuBatch copied_input = router_input;
            auto promise = std::make_shared<
                std::promise<Result<RouterPredictionOutcome>>>();
            std::future<Result<RouterPredictionOutcome>> future =
                promise->get_future();
            const CompiledLayerPlan* next_layer = target.layer;
            const uint32_t prefetch_width = target.prefetch_width;
            submitted = state.router_prediction_worker->try_submit(
                [&model,
                 next_layer,
                 copied_input = std::move(copied_input),
                 prefetch_width,
                 promise]() mutable {
                    try
                    {
                        promise->set_value(run_router_prediction(
                            model,
                            *next_layer,
                            copied_input,
                            prefetch_width));
                    }
                    catch (const std::exception& error)
                    {
                        promise->set_value(Error{
                            ErrorCode::InternalError,
                            std::string("Router prediction failed: ")
                                + error.what()});
                    }
                    catch (...)
                    {
                        promise->set_value(Error{
                            ErrorCode::InternalError,
                            "Router prediction failed"});
                    }
                });
            if (submitted)
            {
                pending.target_layer_id = target.layer->layer_id;
                pending.result = std::move(future);
                ++statistics.expert_route_prediction_async_submissions;
                return {};
            }
        }
        catch (...)
        {
            submitted = false;
        }
        if (!submitted)
            ++statistics.expert_route_prediction_async_fallbacks;
    }

    auto outcome = run_router_prediction(
        model,
        *target.layer,
        router_input,
        target.prefetch_width);
    if (!outcome)
        return outcome.error();
    return apply_router_prediction(
        std::move(outcome).value(),
        state,
        statistics);
}

static Result<void> run_moe(const CompiledModel& model, const MoeBlockPlan& moe, LayerGraphState& layer_state, SessionStatistics& statistics, CpuExpertExecutionScratch& scratch, uint32_t residency_group)
{
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
    expert_team_size = std::min(static_cast<int>(active_expert_count), omp_get_max_threads());
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

    for (const ActiveExpertExecution& active : layer_state.active_experts)
    {
        if (active.failed)
            return active.error;
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
    std::vector<size_t>& backend_indices = scratch.backend_indices;
    std::vector<ExpertBackendRequest>& backend_requests = scratch.backend_requests;
    backend_executed.assign(active_expert_count, 0);
    backend_indices.clear();
    backend_requests.clear();
    backend_indices.reserve(active_expert_count);
    backend_requests.reserve(active_expert_count);
    std::unique_ptr<IExpertBackendBatchSubmission> backend_submission;
    std::chrono::steady_clock::time_point backend_execution_start;
    bool backend_reserved_work = false;
    uint32_t backend_max_token_count = 0;
    uint64_t backend_total_weight_bytes = 0;
    uint64_t backend_accelerated_weight_bytes = 0;
    if (model.expert_backend)
    {
        for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
        {
            ActiveExpertExecution& active = layer_state.active_experts[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            if (expert.gate_up_weight == invalid_tensor_handle
                || (expert.activation != ExpertActivation::GptOssSwiGlu && expert.activation != ExpertActivation::DeepSeekSwiGlu))
            {
                continue;
            }
            backend_indices.push_back(active_index);
            backend_max_token_count = std::max<uint32_t>(backend_max_token_count, static_cast<uint32_t>(active.input.rows()));
            backend_total_weight_bytes += expert.weight_bytes;
            backend_requests.push_back({expert.cache_key, &active.input, &active.output, expert.weight_bytes});
        }
        backend_execution_start = std::chrono::steady_clock::now();
        backend_submission = model.expert_backend->submit_batch(backend_requests);
        if (backend_submission)
        {
            const std::span<const ExpertBackendExecutionResult> planned = backend_submission->planned_results();
            const size_t result_count = std::min(planned.size(), backend_indices.size());
            for (size_t result_index = 0; result_index < result_count; ++result_index)
            {
                if (planned[result_index] != ExpertBackendExecutionResult ::Executed)
                {
                    continue;
                }
                backend_executed[backend_indices[result_index]] = 1;
                const ActiveExpertExecution& active = layer_state.active_experts[backend_indices[result_index]];
                backend_accelerated_weight_bytes += moe.experts[active.batch.expert_id].weight_bytes;
                backend_reserved_work = true;
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
        const TensorData& down = model.weights.at(expert.down_weight);
        if (!model.expert_cache || !gate_up || (!gate_up->mxfp4_file_storage && !down.mxfp4_file_storage))
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
                victim_metadata(model, expert),
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
                if (expert.runtime)
                {
                    if (active.lease.cache_hit)
                        expert.runtime->record_cache_hit();
                    else
                        expert.runtime->record_cache_miss();
                    expert.runtime->set_residency(ExpertCacheState::Resident, TensorLocation::Cpu);
                }
                admit_vulkan_expert(model, expert, active.lease, residency_group, static_cast<uint32_t>(active.input.rows()));
            }
        }
        else
        {
            for (size_t active_index : pending)
            {
                ActiveExpertExecution& active = layer_state.active_experts[active_index];
                const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                const TensorData& down = model.weights.at(expert.down_weight);
                auto requested = model.expert_cache->request_pair(gate_up, down, residency_group, expert.cache_key, victim_metadata(model, expert));
                if (!requested)
                    return requested.error();
                if (expert.runtime)
                {
                    if (requested.value())
                    {
                        expert.runtime->record_cache_hit();
                    }
                    else
                    {
                        expert.runtime->record_cache_miss();
                        expert.runtime->set_residency(ExpertCacheState::Loading, TensorLocation::Automatic);
                    }
                }
            }
        }
    }

    statistics.expert_cache_management_time_microseconds += elapsed_microseconds(cache_management_start);
    compute_wall_time_microseconds += run_experts(model, moe, layer_state, uncached, scratch);

    if (ready_batch_acquired)
    {
        compute_wall_time_microseconds += run_experts(model, moe, layer_state, pending, scratch);
        const auto lease_release_start = std::chrono::steady_clock::now();
        for (size_t active_index : pending)
            layer_state.active_experts[active_index].lease = {};
        pending.clear();
        statistics.expert_cache_management_time_microseconds += elapsed_microseconds(lease_release_start);
    }

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
                victim_metadata(model, expert),
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
            admit_vulkan_expert(model, expert, active.lease, residency_group, static_cast<uint32_t>(active.input.rows()));
            if (expert.runtime)
            {
                expert.runtime->set_residency(ExpertCacheState::Resident, TensorLocation::Cpu);
            }
            ready_indices.push_back(active_index);
        }
        pending.resize(pending_count);
        compute_wall_time_microseconds += run_experts(model, moe, layer_state, ready_indices, scratch);
        for (size_t active_index : ready_indices)
            layer_state.active_experts[active_index].lease = {};
    }

    if (backend_submission)
    {
        const std::vector<ExpertBackendExecutionResult> backend_results = backend_submission->wait();
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
            if (backend_result != ExpertBackendExecutionResult ::Executed)
            {
                backend_executed[active_index] = 0;
                const ActiveExpertExecution& active = layer_state.active_experts[active_index];
                backend_accelerated_weight_bytes -= std::min<uint64_t>(backend_accelerated_weight_bytes, moe.experts[active.batch.expert_id].weight_bytes);
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
            if (expert.runtime)
            {
                expert.runtime->set_residency(ExpertCacheState::Resident, TensorLocation::Vulkan);
            }
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
                if (model.expert_cache && (gate_up.mxfp4_file_storage || down.mxfp4_file_storage))
                {
                    const auto wait_start = std::chrono::steady_clock::now();
                    auto lease = model.expert_cache->acquire_pair(gate_up, down, residency_group, expert.cache_key, victim_metadata(model, expert));
                    active.metrics.cache_wait_time_microseconds += elapsed_microseconds(wait_start);
                    if (!lease)
                    {
                        if (expert.runtime)
                        {
                            expert.runtime->set_residency(ExpertCacheState::Failed, TensorLocation::Automatic);
                        }
                        return lease.error();
                    }
                    active.lease = std::move(lease).value();
                    admit_vulkan_expert(model, expert, active.lease, residency_group, static_cast<uint32_t>(active.input.rows()));
                    if (expert.runtime)
                    {
                        expert.runtime->set_residency(ExpertCacheState::Resident, TensorLocation::Cpu);
                    }
                }
            }
            statistics.expert_cache_management_time_microseconds += elapsed_microseconds(fallback_cache_start);
            compute_wall_time_microseconds += run_experts(model, moe, layer_state, failed_indices, scratch);
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

static Result<void> execute_speculative_layer(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    uint64_t position_offset,
    CpuLayerCache& cache,
    CpuBatch& hidden,
    SessionStatistics& statistics,
    CpuExpertExecutionScratch& scratch,
    CpuAttentionExecutionScratch& attention_scratch)
{
    LayerGraphState layer_state;
    const uint32_t multiplier = model.descriptor.hyper_connection_multiplier;
    const auto attention_start = std::chrono::steady_clock::now();
    CpuHyperConnectionMix attention_mix;
    const CpuBatch* attention_input = &hidden;
    if (multiplier > 1)
    {
        auto mixed = hyper_connection_pre(hidden, model.weights.at(layer.hyper_connection.attention_function), model.weights.at(layer.hyper_connection.attention_scale),
                                          model.weights.at(layer.hyper_connection.attention_base), multiplier, model.descriptor.hyper_connection_iterations,
                                          model.descriptor.norm_epsilon, model.descriptor.hyper_connection_epsilon);
        if (!mixed)
            return mixed.error();
        attention_mix = std::move(mixed).value();
        attention_input = &attention_mix.reduced;
    }
    if (model.speculative.kind == SpeculativeModelKind::Mtp)
    {
        execute_attention_block_into(
            model.weights,
            layer.attention,
            model.descriptor.norm_epsilon,
            model.descriptor.kv_cache_dtype,
            position_offset,
            cache,
            attention_scratch,
            *attention_input,
            attention_scratch.output);
        hidden.swap(attention_scratch.output);
    }
    else
    {
        auto attention = execute_dspark_attention(model.weights, layer.attention, model.descriptor.norm_epsilon, position_offset, cache, *attention_input);
        if (!attention)
            return attention.error();
        if (multiplier > 1)
        {
            auto connected = hyper_connection_post(attention.value(), hidden, attention_mix, multiplier);
            if (!connected)
                return connected.error();
            hidden = std::move(connected).value();
        }
        else
        {
            add_batch_inplace(hidden, attention.value());
        }
    }
    statistics.attention_time_microseconds += elapsed_microseconds(attention_start);

    const MoeBlockPlan& moe = layer.moe;
    layer_state.router_start = std::chrono::steady_clock::now();
    if (multiplier > 1)
    {
        auto mixed = hyper_connection_pre(hidden, model.weights.at(layer.hyper_connection.ffn_function), model.weights.at(layer.hyper_connection.ffn_scale),
                                          model.weights.at(layer.hyper_connection.ffn_base), multiplier, model.descriptor.hyper_connection_iterations,
                                          model.descriptor.norm_epsilon, model.descriptor.hyper_connection_epsilon);
        if (!mixed)
            return mixed.error();
        layer_state.ffn_hyper_mix = std::move(mixed).value();
        rms_norm_batch_into(layer_state.ffn_hyper_mix.reduced, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset);
    }
    else
    {
        rms_norm_batch_into(hidden, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset);
    }
    linear_batch_into(model.weights.at(moe.router_weight), layer_state.normalized, layer_state.router_logits);
    ExpertDispatchOptions dispatch_options;
    dispatch_options.expert_count = static_cast<uint32_t>(moe.experts.size());
    dispatch_options.top_k = moe.top_k;
    dispatch_options.score_function = moe.score_function;
    dispatch_options.normalization = moe.normalization;
    dispatch_options.routed_scaling_factor = moe.routed_scaling_factor;
    if (moe.router_selection_bias != invalid_tensor_handle)
    {
        dispatch_options.selection_bias = model.weights.at(moe.router_selection_bias).float32_values();
    }
    dispatch_options.flags = has_flag(moe.flags, MoeBlockNormalizeTopKWeights)
                                 ? static_cast<uint32_t>(ExpertDispatchNormalizeTopKWeights)
                                 : 0u;
    ExpertDispatcher dispatcher;
    auto dispatched = dispatcher.dispatch_into(layer_state.router_logits.values(), static_cast<uint32_t>(layer_state.router_logits.rows()), dispatch_options, layer_state.dispatch_plan);
    if (!dispatched)
        return dispatched.error();
    statistics.expert_assignments += layer_state.dispatch_plan.assignment_count;
    layer_state.active_experts.resize(layer_state.dispatch_plan.batches.size());
    for (size_t batch_index = 0; batch_index < layer_state.dispatch_plan.batches.size(); ++batch_index)
    {
        const ExpertBatch& batch = layer_state.dispatch_plan.batches[batch_index];
        statistics.expert_token_counts[batch.expert_id] += batch.routes.size();
        record_expert_weight_demand(moe.experts[batch.expert_id], batch.routes.size(), statistics);
        layer_state.active_experts[batch_index].prepare(batch);
    }
    layer_state.router_logits.clear();
    statistics.router_time_microseconds += elapsed_microseconds(layer_state.router_start);
    layer_state.expert_start = std::chrono::steady_clock::now();

    std::future<OverlappedSharedExpertResult> shared_expert;
    if (can_overlap_vulkan_shared_expert(model, moe))
    {
        shared_expert = std::async(std::launch::async, [&model, &moe, &layer_state] {
            return run_overlapped_shared_expert(model, moe, layer_state.normalized);
        });
    }
    const auto expert_engine_start = std::chrono::steady_clock::now();
    auto executed = run_moe(model, moe, layer_state, statistics, scratch, layer.layer_id);
    statistics.expert_engine_time_microseconds += elapsed_microseconds(expert_engine_start);
    if (!executed)
        return executed.error();
    if (shared_expert.valid())
    {
        OverlappedSharedExpertResult result = shared_expert.get();
        layer_state.shared_expert_output = std::move(result.output);
        record_captured_vulkan_execution_delta(statistics, result.vulkan);
    }

    const auto combine_start = std::chrono::steady_clock::now();
    if (moe.has_shared_expert && layer_state.shared_expert_output.rows() == 0)
    {
        ExpertExecutionMetrics shared_metrics;
        layer_state.shared_expert_output = run_shared_expert(model, moe, layer_state.normalized, shared_metrics);
    }
    CpuBatch& moe_output = layer_state.normalized;
    moe_output.reset(hidden.rows(), model.descriptor.hidden_size, true);
    for (const ActiveExpertExecution& active : layer_state.active_experts)
    {
        for (size_t batch_index = 0; batch_index < active.batch.routes.size(); ++batch_index)
        {
            const ExpertRoute& route = active.batch.routes[batch_index];
            float* destination = moe_output.row(route.token_index);
            const float* source = active.output.row(batch_index);
            for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
            {
                destination[column] += route.weight * source[column];
            }
        }
    }
    if (moe.has_shared_expert)
        add_batch_inplace(moe_output, layer_state.shared_expert_output);
    if (multiplier > 1)
    {
        auto connected = hyper_connection_post(moe_output, hidden, layer_state.ffn_hyper_mix, multiplier);
        if (!connected)
            return connected.error();
        hidden = std::move(connected).value();
    }
    else
    {
        add_batch_inplace(hidden, moe_output);
    }
    statistics.expert_combine_time_microseconds += elapsed_microseconds(combine_start);
    statistics.expert_time_microseconds += elapsed_microseconds(layer_state.expert_start);
    return {};
}

static Result<CpuBatch> prepare_mtp_hidden(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    const CpuBatch& target_hidden)
{
    if (input_ids.empty()
        || input_ids.size() != target_hidden.rows()
        || target_hidden.columns() != model.descriptor.hidden_size)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid Qwen MTP input batch"};
    }

    CpuBatch embeddings;
    embedding_batch_into(
        model.weights.at(model.token_embedding),
        input_ids,
        embeddings);
    CpuBatch normalized_embeddings = rms_norm_batch(
        embeddings,
        model.weights.at(
            model.speculative.mtp_embedding_norm_weight),
        model.descriptor.norm_epsilon,
        model.descriptor.norm_weight_offset);
    CpuBatch normalized_hidden = rms_norm_batch(
        target_hidden,
        model.weights.at(model.speculative.mtp_hidden_norm_weight),
        model.descriptor.norm_epsilon,
        model.descriptor.norm_weight_offset);
    CpuBatch packed(
        target_hidden.rows(),
        model.descriptor.hidden_size * 2);
    for (size_t row = 0; row < target_hidden.rows(); ++row)
    {
        std::copy_n(
            normalized_embeddings.row(row),
            model.descriptor.hidden_size,
            packed.row(row));
        std::copy_n(
            normalized_hidden.row(row),
            model.descriptor.hidden_size,
            packed.row(row) + model.descriptor.hidden_size);
    }
    return linear_batch(
        model.weights.at(
            model.speculative.mtp_input_projection_weight),
        packed);
}

static Result<CpuBatch> execute_mtp_batch(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    const CpuBatch& target_hidden,
    uint64_t position_offset,
    SessionStatistics& statistics,
    CpuSessionState& state)
{
    if (model.speculative.layers.size() != 1
        || state.speculative_layers.size() != 1)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid Qwen MTP execution state"};
    }
    auto prepared = prepare_mtp_hidden(
        model,
        input_ids,
        target_hidden);
    if (!prepared)
        return prepared.error();
    CpuBatch hidden = std::move(prepared).value();
    auto executed = execute_speculative_layer(
        model,
        model.speculative.layers.front(),
        position_offset,
        state.speculative_layers.front(),
        hidden,
        statistics,
        state.expert_scratch,
        state.attention_scratch);
    if (!executed)
        return executed.error();
    return rms_norm_batch(
        hidden,
        model.weights.at(model.speculative.final_norm_weight),
        model.descriptor.norm_epsilon,
        model.descriptor.norm_weight_offset);
}

static Result<void> append_mtp_context(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    const CpuBatch& target_hidden,
    uint64_t position_offset,
    CpuSessionState& state)
{
    if (model.speculative.layers.size() != 1
        || state.speculative_layers.size() != 1)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid Qwen MTP context state"};
    }
    auto prepared = prepare_mtp_hidden(
        model,
        input_ids,
        target_hidden);
    if (!prepared)
        return prepared.error();
    append_attention_context_into(
        model.weights,
        model.speculative.layers.front().attention,
        model.descriptor.norm_epsilon,
        model.descriptor.kv_cache_dtype,
        position_offset,
        state.speculative_layers.front(),
        state.attention_scratch,
        prepared.value());
    return {};
}

static Result<void> update_mtp_context(
    const CompiledModel& model,
    CpuSessionState& state)
{
    const CpuBatch& target_hidden = state.speculative_main_hidden;
    if (target_hidden.rows() == 0
        || target_hidden.columns() != model.descriptor.hidden_size)
    {
        return Error{
            ErrorCode::InternalError,
            "target execution did not capture Qwen final hidden states"};
    }

    const bool direct_alignment =
        !state.speculative_direct_alignment_ids.empty();
    const std::vector<int32_t>& input_ids =
        direct_alignment
            ? state.speculative_direct_alignment_ids
            : state.speculative_input_ids;
    if (input_ids.size() != target_hidden.rows())
    {
        return Error{
            ErrorCode::InternalError,
            "Qwen MTP alignment IDs do not match target hidden states"};
    }
    if (direct_alignment
        && state.mtp_pending_target_hidden.rows() != 0)
    {
        return Error{
            ErrorCode::InternalError,
            "Qwen MTP direct alignment still has a pending target row"};
    }

    const bool has_pending =
        state.mtp_pending_target_hidden.rows() != 0;
    if (has_pending
        && (state.mtp_pending_target_hidden.rows() != 1
            || state.mtp_pending_target_hidden.columns()
                   != model.descriptor.hidden_size
            || state.mtp_pending_target_position + 1
                   != state.speculative_main_hidden_position))
    {
        return Error{
            ErrorCode::InternalError,
            "Qwen MTP pending target row is out of sequence"};
    }

    const size_t aligned_rows =
        direct_alignment
            ? target_hidden.rows() - 1
            : (has_pending ? 1 : 0) + target_hidden.rows() - 1;
    if (aligned_rows != 0)
    {
        CpuBatch aligned_hidden(
            aligned_rows,
            model.descriptor.hidden_size);
        std::vector<int32_t> aligned_ids;
        aligned_ids.reserve(aligned_rows);
        size_t output_row = 0;
        uint64_t aligned_position =
            state.speculative_main_hidden_position;
        if (!direct_alignment && has_pending)
        {
            aligned_position = state.mtp_pending_target_position;
            std::copy_n(
                state.mtp_pending_target_hidden.row(0),
                model.descriptor.hidden_size,
                aligned_hidden.row(output_row++));
            aligned_ids.push_back(input_ids.front());
        }
        for (size_t row = 0;
             row + 1 < target_hidden.rows();
             ++row)
        {
            std::copy_n(
                target_hidden.row(row),
                model.descriptor.hidden_size,
                aligned_hidden.row(output_row++));
            aligned_ids.push_back(
                input_ids[row + (direct_alignment ? 0 : 1)]);
        }
        auto aligned = append_mtp_context(
            model,
            aligned_ids,
            aligned_hidden,
            aligned_position,
            state);
        if (!aligned)
            return aligned.error();
    }

    state.mtp_pending_target_hidden.reset(
        1,
        model.descriptor.hidden_size,
        false);
    std::copy_n(
        target_hidden.row(target_hidden.rows() - 1),
        model.descriptor.hidden_size,
        state.mtp_pending_target_hidden.row(0));
    state.mtp_pending_target_position =
        state.speculative_main_hidden_position
        + target_hidden.rows() - 1;
    state.speculative_input_ids.clear();
    state.speculative_direct_alignment_ids.clear();
    return {};
}

Result<void> CpuExecutor::update_speculative_context(const CompiledModel& model, SessionStatistics& statistics, CpuSessionState& state) const
{
    const OpenMpHybridTeamLimit team_limit(model.hybrid_mode);
    if (!model.speculative.enabled()
        || !state.speculative_context_enabled)
        return {};
    const auto started = std::chrono::steady_clock::now();
    const VulkanExecutionSnapshot vulkan_before =
        capture_vulkan_execution();
    if (model.speculative.kind == SpeculativeModelKind::Mtp)
    {
        auto updated = update_mtp_context(
            model,
            state);
        if (!updated)
            return updated.error();
        record_vulkan_execution_delta(statistics, vulkan_before);
        statistics.speculative_context_time_microseconds +=
            elapsed_microseconds(started);
        return {};
    }
    const uint32_t expected_columns = model.descriptor.hidden_size
                                      * static_cast<uint32_t>(model.speculative.target_layer_ids.size());
    if (state.speculative_main_hidden.rows() == 0
        || state.speculative_main_hidden.columns() != expected_columns)
    {
        return Error{
            ErrorCode::InternalError,
            "target execution did not capture DSpark context features"};
    }
    CpuBatch projected = linear_batch(model.weights.at(model.speculative.main_projection_weight), state.speculative_main_hidden);
    projected = rms_norm_batch(projected, model.weights.at(model.speculative.main_norm_weight), model.descriptor.norm_epsilon, model.descriptor.norm_weight_offset);
    if (state.speculative_layers.size() != model.speculative.layers.size())
        state.speculative_layers.resize(model.speculative.layers.size());
    for (size_t layer_index = 0; layer_index < model.speculative.layers.size(); ++layer_index)
    {
        auto appended = append_dspark_attention_context(model.weights, model.speculative.layers[layer_index].attention, model.descriptor.norm_epsilon,
                                                        state.speculative_main_hidden_position, state.speculative_layers[layer_index], projected);
        if (!appended)
            return appended.error();
    }
    record_vulkan_execution_delta(statistics, vulkan_before);
    statistics.speculative_context_time_microseconds += elapsed_microseconds(started);
    return {};
}

static Result<CpuSpeculativeProposal> propose_mtp(
    const CompiledModel& model,
    int32_t input_id,
    SessionStatistics& statistics,
    CpuSessionState& state,
    uint64_t position_offset,
    const CpuSpeculativeSampler& sampler)
{
    if (!sampler)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "Qwen MTP proposal requires a token sampler"};
    }
    if (input_id < 0
        || static_cast<uint32_t>(input_id)
               >= model.descriptor.vocabulary_size
        || state.speculative_layers.size() != 1
        || state.mtp_pending_target_hidden.rows() != 1
        || state.mtp_pending_target_hidden.columns()
               != model.descriptor.hidden_size
        || position_offset == 0
        || state.mtp_pending_target_position + 1
               != position_offset)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid Qwen MTP proposal state"};
    }

    const auto started = std::chrono::steady_clock::now();
    const VulkanExecutionSnapshot vulkan_before =
        capture_vulkan_execution();
    ExpertCacheStatistics cache_before;
    if (model.expert_cache)
        cache_before = model.expert_cache->statistics();
    ExpertBackendStatistics backend_before;
    if (model.expert_backend)
        backend_before = model.expert_backend->statistics();

    CpuSpeculativeProposal proposal;
    proposal.token_ids.reserve(model.speculative.block_size);
    proposal.logits.reserve(model.speculative.block_size);
    proposal.confidence_logits.reserve(
        model.speculative.block_size);
    proposal.committed_context_rows = 1;
    CpuBatch previous_hidden =
        state.mtp_pending_target_hidden;
    int32_t previous_token = input_id;
    for (uint32_t row = 0;
         row < model.speculative.block_size;
         ++row)
    {
        const std::span<const int32_t> token(
            &previous_token,
            1);
        auto mtp_hidden = execute_mtp_batch(
            model,
            token,
            previous_hidden,
            state.mtp_pending_target_position + row,
            statistics,
            state);
        if (!mtp_hidden)
            return mtp_hidden.error();
        CpuBatch logits = linear_batch(
            model.weights.at(model.lm_head_weight),
            mtp_hidden.value());
        std::vector<float> row_logits(
            logits.row(0),
            logits.row(0) + logits.columns());
        auto sampled = sampler(row_logits);
        if (!sampled)
            return sampled.error();
        previous_token = sampled.value();
        proposal.token_ids.push_back(previous_token);
        proposal.logits.push_back(std::move(row_logits));
        proposal.confidence_logits.push_back(
            std::numeric_limits<float>::infinity());
        previous_hidden = std::move(mtp_hidden).value();
    }
    state.mtp_pending_target_hidden.clear();

    if (model.expert_cache)
    {
        const ExpertCacheStatistics cache_after =
            model.expert_cache->statistics();
        statistics.expert_cache_hits +=
            cache_after.hits - cache_before.hits;
        statistics.expert_cache_misses +=
            cache_after.misses - cache_before.misses;
        statistics.expert_cache_evictions +=
            cache_after.evictions - cache_before.evictions;
        statistics.expert_cache_bytes_read +=
            cache_after.bytes_read - cache_before.bytes_read;
        statistics.expert_cache_resident_bytes =
            cache_after.resident_bytes;
    }
    if (model.expert_backend)
    {
        record_expert_backend_delta(
            statistics,
            backend_before,
            model.expert_backend->statistics());
    }
    record_vulkan_execution_delta(statistics, vulkan_before);
    ++statistics.speculative_proposals;
    statistics.speculative_draft_tokens +=
        proposal.token_ids.size();
    statistics.speculative_draft_time_microseconds +=
        elapsed_microseconds(started);
    return proposal;
}

Result<CpuSpeculativeProposal> CpuExecutor::propose_speculative(const CompiledModel& model, int32_t input_id, SessionStatistics& statistics, CpuSessionState& state, uint64_t position_offset,
                                                                const CpuSpeculativeSampler& sampler) const
{
    const OpenMpHybridTeamLimit team_limit(model.hybrid_mode);
    if (!model.speculative.enabled())
    {
        return Error{
            ErrorCode::UnsupportedModel,
            "the model does not provide a speculative execution plan"};
    }
    if (model.speculative.kind == SpeculativeModelKind::Mtp)
    {
        return propose_mtp(
            model,
            input_id,
            statistics,
            state,
            position_offset,
            sampler);
    }
    if (!sampler)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "DSpark proposal requires a token sampler"};
    }
    if (input_id < 0
        || static_cast<uint32_t>(input_id)
               >= model.descriptor.vocabulary_size
        || state.speculative_layers.size()
               != model.speculative.layers.size())
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid DSpark proposal state"};
    }
    for (const CpuLayerCache& cache : state.speculative_layers)
    {
        if (cache.latent_token_count != position_offset)
        {
            return Error{
                ErrorCode::InternalError,
                "DSpark context cache is out of sync with the target model"};
        }
    }

    const auto started = std::chrono::steady_clock::now();
    const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution();
    ExpertCacheStatistics cache_before;
    if (model.expert_cache)
        cache_before = model.expert_cache->statistics();
    ExpertBackendStatistics backend_before;
    if (model.expert_backend)
        backend_before = model.expert_backend->statistics();

    std::vector<int32_t> draft_input_ids(model.speculative.block_size, static_cast<int32_t>(model.speculative.noise_token_id));
    draft_input_ids.front() = input_id;
    CpuBatch hidden;
    embedding_batch_into(model.weights.at(model.token_embedding), draft_input_ids, hidden);
    expand_hyper_connections(hidden, model.descriptor.hyper_connection_multiplier, state.expert_scratch.staged_output);
    for (size_t layer_index = 0; layer_index < model.speculative.layers.size(); ++layer_index)
    {
        auto executed = execute_speculative_layer(
            model,
            model.speculative.layers[layer_index],
            position_offset,
            state.speculative_layers[layer_index],
            hidden,
            statistics,
            state.expert_scratch,
            state.attention_scratch);
        if (!executed)
            return executed.error();
    }

    auto headed = hyper_connection_head(hidden, model.weights.at(model.speculative.hyper_head_function), model.weights.at(model.speculative.hyper_head_scale),
                                        model.weights.at(model.speculative.hyper_head_base), model.descriptor.hyper_connection_multiplier, model.descriptor.norm_epsilon,
                                        model.descriptor.hyper_connection_epsilon);
    if (!headed)
        return headed.error();
    CpuBatch head_hidden = std::move(headed).value();
    CpuBatch normalized = rms_norm_batch(head_hidden, model.weights.at(model.speculative.final_norm_weight), model.descriptor.norm_epsilon, model.descriptor.norm_weight_offset);
    CpuBatch base_logits = linear_batch(model.weights.at(model.lm_head_weight), normalized);

    CpuSpeculativeProposal proposal;
    proposal.token_ids.reserve(model.speculative.block_size);
    proposal.logits.reserve(model.speculative.block_size);
    proposal.confidence_logits.reserve(model.speculative.block_size);
    int32_t previous_token = input_id;
    const TensorData& confidence = model.weights.at(model.speculative.confidence_weight);
    for (uint32_t row = 0; row < model.speculative.block_size; ++row)
    {
        const std::span<const int32_t> previous(&previous_token, 1);
        CpuBatch markov_embedding;
        embedding_batch_into(model.weights.at(model.speculative.markov_embedding_weight), previous, markov_embedding);
        CpuBatch markov_logits = linear_batch(model.weights.at(model.speculative.markov_head_weight), markov_embedding);
        std::vector<float> row_logits(model.descriptor.vocabulary_size);
        for (uint32_t token_id = 0; token_id < model.descriptor.vocabulary_size; ++token_id)
        {
            row_logits[token_id] = base_logits.row(row)[token_id]
                                   + markov_logits.row(0)[token_id];
        }
        auto sampled = sampler(row_logits);
        if (!sampled)
            return sampled.error();
        const int32_t selected = sampled.value();
        proposal.token_ids.push_back(selected);
        proposal.logits.push_back(std::move(row_logits));
        previous_token = selected;

        float confidence_logit = 0.0f;
        const std::span<const uint16_t> confidence_values = confidence.bfloat16_values();
        for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
        {
            confidence_logit += head_hidden.row(row)[column]
                                * bfloat16_to_float(confidence_values[column]);
        }
        for (uint32_t column = 0; column < model.speculative.markov_rank; ++column)
        {
            confidence_logit += markov_embedding.row(0)[column] * bfloat16_to_float(confidence_values[model.descriptor.hidden_size + column]);
        }
        proposal.confidence_logits.push_back(confidence_logit);
    }

    if (model.expert_cache)
    {
        const ExpertCacheStatistics cache_after = model.expert_cache->statistics();
        statistics.expert_cache_hits += cache_after.hits - cache_before.hits;
        statistics.expert_cache_misses += cache_after.misses - cache_before.misses;
        statistics.expert_cache_evictions += cache_after.evictions - cache_before.evictions;
        statistics.expert_cache_bytes_read += cache_after.bytes_read - cache_before.bytes_read;
        statistics.expert_cache_resident_bytes = cache_after.resident_bytes;
    }
    if (model.expert_backend)
    {
        record_expert_backend_delta(statistics, backend_before, model.expert_backend->statistics());
    }
    record_vulkan_execution_delta(statistics, vulkan_before);
    ++statistics.speculative_proposals;
    statistics.speculative_draft_tokens += proposal.token_ids.size();
    statistics.speculative_draft_time_microseconds += elapsed_microseconds(started);
    return proposal;
}

Result<std::vector<std::vector<float>>> CpuExecutor::execute(const CompiledModel& model, std::span<const int32_t> input_ids, SessionStatistics& statistics, CpuSessionState& state, uint64_t position_offset) const
{
    const OpenMpHybridTeamLimit team_limit(model.hybrid_mode);
    const VulkanExecutionSnapshot initial_vulkan_execution = capture_vulkan_execution();
    for (int32_t token_id : input_ids)
    {
        if (token_id < 0 || static_cast<uint32_t>(token_id) >= model.descriptor.vocabulary_size)
            return Error{ErrorCode::InvalidArgument, "token id is outside the model vocabulary"};
    }

    if (statistics.expert_token_counts.size() < model.descriptor.expert_count)
        statistics.expert_token_counts.resize(model.descriptor.expert_count, 0);

    if (state.layers.size() != model.layers.size())
        state.layers.resize(model.layers.size());
    if (state.execution_layers.size() != model.layers.size())
    {
        state.execution_layers.resize(model.layers.size());
    }
    for (LayerGraphState& layer_state : state.execution_layers)
    {
        layer_state.reset();
    }

    ExpertCacheStatistics execution_cache_before;
    if (model.expert_cache)
        execution_cache_before = model.expert_cache->statistics();
    ExpertBackendStatistics expert_backend_before;
    if (model.expert_backend)
    {
        expert_backend_before = model.expert_backend->statistics();
    }

    CpuBatch& hidden = state.hidden;
    hidden.clear();
    if (model.speculative.enabled()
        && state.speculative_context_enabled)
    {
        const uint32_t speculative_hidden_columns =
            model.speculative.kind == SpeculativeModelKind::Mtp
                ? model.descriptor.hidden_size
                : model.descriptor.hidden_size
                      * static_cast<uint32_t>(
                          model.speculative.target_layer_ids.size());
        state.speculative_main_hidden.reset(
            input_ids.size(),
            speculative_hidden_columns,
            true);
        state.speculative_main_hidden_position = position_offset;
        if (state.speculative_layers.size() != model.speculative.layers.size())
            state.speculative_layers.resize(model.speculative.layers.size());
        if (model.speculative.kind == SpeculativeModelKind::Mtp)
        {
            state.speculative_input_ids.assign(
                input_ids.begin(),
                input_ids.end());
            state.speculative_direct_alignment_ids.clear();
        }
    }
    else
    {
        state.speculative_main_hidden.clear();
    }
    std::vector<std::vector<float>> logits;
    std::vector<LayerGraphState>& layer_states = state.execution_layers;
    ExpertDispatcher dispatcher;
    PendingRouterPrediction pending_router_prediction;
    for (const ExecutionWave& wave : model.schedule.waves)
    {
        for (ExecutionNodeId node_id : wave.nodes)
        {
            const ExecutionNode* node = model.graph.find(node_id);
            if (!node)
                return Error{ErrorCode::InternalError, "execution schedule references an invalid node"};

            if (node->type == ExecutionNodeType::TokenEmbedding)
            {
                const auto embedding_start = std::chrono::steady_clock::now();
                embedding_batch_into(model.weights.at(model.token_embedding), input_ids, hidden);
                expand_hyper_connections(hidden, model.descriptor.hyper_connection_multiplier, state.expert_scratch.staged_output);
                statistics.embedding_time_microseconds += elapsed_microseconds(embedding_start);
                continue;
            }
            if (node->type == ExecutionNodeType::FinalNorm)
            {
                const auto final_norm_start = std::chrono::steady_clock::now();
                if (model.descriptor.hyper_connection_multiplier > 1)
                {
                    auto head = hyper_connection_head(
                        hidden,
                        model.weights.at(model.hyper_head_function),
                        model.weights.at(model.hyper_head_scale),
                        model.weights.at(model.hyper_head_base),
                        model.descriptor.hyper_connection_multiplier,
                        model.descriptor.norm_epsilon,
                        model.descriptor.hyper_connection_epsilon);
                    if (!head)
                        return head.error();
                    hidden = std::move(head).value();
                }
                rms_norm_batch_into(hidden, model.weights.at(model.final_norm_weight), model.descriptor.norm_epsilon, state.expert_scratch.staged_output, model.descriptor.norm_weight_offset);
                hidden.swap(state.expert_scratch.staged_output);
                if (model.speculative.kind == SpeculativeModelKind::Mtp
                    && state.speculative_context_enabled)
                {
                    for (size_t row = 0; row < hidden.rows(); ++row)
                    {
                        std::copy_n(
                            hidden.row(row),
                            model.descriptor.hidden_size,
                            state.speculative_main_hidden.row(row));
                    }
                }
                statistics.final_norm_time_microseconds += elapsed_microseconds(final_norm_start);
                continue;
            }
            if (node->type == ExecutionNodeType::LmHead)
            {
                const auto lm_head_start = std::chrono::steady_clock::now();
                logits = batch_to_vectors(linear_batch(model.weights.at(model.lm_head_weight), hidden));
                statistics.lm_head_time_microseconds += elapsed_microseconds(lm_head_start);
                continue;
            }
            if (node->layer_id >= model.layers.size())
                return Error{ErrorCode::InternalError, "execution node layer is out of range"};

            const CompiledLayerPlan& layer = model.layers[node->layer_id];
            LayerGraphState& layer_state = layer_states[node->layer_id];
            const MoeBlockPlan& moe = layer.moe;
            if (node->type == ExecutionNodeType::Attention)
            {
                const auto attention_start = std::chrono::steady_clock::now();
                if (has_flag(layer.attention.flags, AttentionBlockGatedDeltaNet))
                {
                    execute_gated_delta_net_into(
                        model.weights,
                        layer.attention,
                        model.descriptor.norm_epsilon,
                        state.layers[layer.layer_id],
                        state.gated_delta_scratch,
                        hidden,
                        state.gated_delta_scratch.output);
                    hidden.swap(state.gated_delta_scratch.output);
                }
                else if (has_flag(layer.attention.flags, AttentionBlockLatent))
                {
                    CpuHyperConnectionMix hyper_mix;
                    const CpuBatch* attention_input = &hidden;
                    if (model.descriptor.hyper_connection_multiplier > 1)
                    {
                        auto mixed = hyper_connection_pre(
                            hidden,
                            model.weights.at(layer.hyper_connection.attention_function),
                            model.weights.at(layer.hyper_connection.attention_scale),
                            model.weights.at(layer.hyper_connection.attention_base),
                            model.descriptor.hyper_connection_multiplier,
                            model.descriptor.hyper_connection_iterations,
                            model.descriptor.norm_epsilon,
                            model.descriptor.hyper_connection_epsilon);
                        if (!mixed)
                            return mixed.error();
                        hyper_mix = std::move(mixed).value();
                        attention_input = &hyper_mix.reduced;
                    }
                    auto output = execute_latent_attention(
                        model.weights,
                        layer.attention,
                        model.descriptor.norm_epsilon,
                        position_offset,
                        state.layers[layer.layer_id],
                        *attention_input);
                    if (!output)
                        return output.error();
                    if (model.descriptor.hyper_connection_multiplier > 1)
                    {
                        auto connected = hyper_connection_post(output.value(), hidden, hyper_mix, model.descriptor.hyper_connection_multiplier);
                        if (!connected)
                            return connected.error();
                        hidden = std::move(connected).value();
                    }
                    else
                    {
                        add_batch_inplace(hidden, output.value());
                    }
                }
                else
                {
                    execute_attention_block_into(model.weights, layer.attention, model.descriptor.norm_epsilon, model.descriptor.kv_cache_dtype, position_offset, state.layers[layer.layer_id], state.attention_scratch, hidden,
                                                 state.attention_scratch.output);
                    hidden.swap(state.attention_scratch.output);
                }
                statistics.attention_time_microseconds += elapsed_microseconds(attention_start);
                continue;
            }
            if (node->type == ExecutionNodeType::Router)
            {
                auto completed_prediction = complete_router_prediction(
                    layer.layer_id,
                    pending_router_prediction,
                    state,
                    statistics);
                if (!completed_prediction)
                    return completed_prediction.error();
                layer_state.router_start = std::chrono::steady_clock::now();
                if (model.descriptor.hyper_connection_multiplier > 1)
                {
                    auto mixed = hyper_connection_pre(
                        hidden,
                        model.weights.at(layer.hyper_connection.ffn_function),
                        model.weights.at(layer.hyper_connection.ffn_scale),
                        model.weights.at(layer.hyper_connection.ffn_base),
                        model.descriptor.hyper_connection_multiplier,
                        model.descriptor.hyper_connection_iterations,
                        model.descriptor.norm_epsilon,
                        model.descriptor.hyper_connection_epsilon);
                    if (!mixed)
                        return mixed.error();
                    layer_state.ffn_hyper_mix = std::move(mixed).value();
                    rms_norm_batch_into(layer_state.ffn_hyper_mix.reduced, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset);
                }
                else
                {
                    rms_norm_batch_into(hidden, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset);
                }
                auto predicted = predict_next_router_routes(
                    model,
                    layer,
                    layer_state.normalized,
                    state,
                    statistics,
                    pending_router_prediction);
                if (!predicted)
                    return predicted.error();
                linear_batch_into(model.weights.at(moe.router_weight), layer_state.normalized, layer_state.router_logits);
                if (moe.router_bias != invalid_tensor_handle)
                {
                    add_bias_inplace(layer_state.router_logits, model.weights.at(moe.router_bias));
                }
                continue;
            }
            if (node->type == ExecutionNodeType::ExpertDispatch)
            {
                ExpertDispatchOptions options;
                options.expert_count = static_cast<uint32_t>(moe.experts.size());
                options.top_k = moe.top_k;
                options.score_function = moe.score_function;
                options.normalization = moe.normalization;
                options.routed_scaling_factor = moe.routed_scaling_factor;
                if (moe.router_selection_bias != invalid_tensor_handle)
                    options.selection_bias = model.weights.at(moe.router_selection_bias).float32_values();
                std::vector<uint32_t> explicit_expert_ids;
                if (moe.token_experts != invalid_tensor_handle)
                {
                    const std::span<const int64_t> table = model.weights.at(moe.token_experts).int64_values();
                    explicit_expert_ids.resize(input_ids.size() * moe.top_k);
                    for (size_t token_index = 0; token_index < input_ids.size(); ++token_index)
                    {
                        for (uint32_t route = 0; route < moe.top_k; ++route)
                            explicit_expert_ids[token_index * moe.top_k + route] = static_cast<uint32_t>(table[static_cast<size_t>(input_ids[token_index]) * moe.top_k + route]);
                    }
                    options.explicit_expert_ids = explicit_expert_ids;
                }
                options.flags = 0;
                if (has_flag(moe.flags, MoeBlockNormalizeTopKWeights))
                {
                    options.flags |= ExpertDispatchNormalizeTopKWeights;
                }
                auto dispatched = dispatcher.dispatch_into(layer_state.router_logits.values(), static_cast<uint32_t>(layer_state.router_logits.rows()), options, layer_state.dispatch_plan);
                if (!dispatched)
                    return dispatched.error();

                const ExpertDispatchPlan& plan = layer_state.dispatch_plan;
                statistics.expert_assignments += static_cast<uint64_t>(plan.assignment_count);
                layer_state.active_experts.resize(plan.batches.size());
                if (hidden.rows() == 1 && layer.layer_id < state.layers.size())
                    resolve_router_predictions(model, layer, plan, state, statistics, true);
                for (size_t batch_index = 0; batch_index < plan.batches.size(); ++batch_index)
                {
                    const ExpertBatch& batch = plan.batches[batch_index];
                    statistics.expert_token_counts[batch.expert_id] += static_cast<uint64_t>(batch.routes.size());
                    const ExpertPlan& expert = moe.experts[batch.expert_id];
                    record_expert_weight_demand(expert, batch.routes.size(), statistics);
                    if (expert.runtime)
                    {
                        expert.runtime->record_dispatch(batch.routes.size(), statistics.expert_assignments);
                    }
                    ActiveExpertExecution& active = layer_state.active_experts[batch_index];
                    active.prepare(batch);
                }
                layer_state.router_logits.clear();
                statistics.router_time_microseconds += elapsed_microseconds(layer_state.router_start);
                layer_state.expert_start = std::chrono::steady_clock::now();

                continue;
            }
            if (node->type == ExecutionNodeType::Expert || node->type == ExecutionNodeType::ExpertGroup)
            {
                if (layer_state.experts_executed)
                    continue;
                std::future<OverlappedSharedExpertResult> shared_expert;
                if (can_overlap_vulkan_shared_expert(model, moe))
                {
                    shared_expert = std::async(std::launch::async, [&model, &moe, &layer_state] {
                        return run_overlapped_shared_expert(model, moe, layer_state.normalized);
                    });
                }
                const auto expert_engine_start = std::chrono::steady_clock::now();
                auto executed = run_moe(model, moe, layer_state, statistics, state.expert_scratch, layer.layer_id);
                statistics.expert_engine_time_microseconds += elapsed_microseconds(expert_engine_start);
                if (!executed)
                    return executed.error();
                if (shared_expert.valid())
                {
                    OverlappedSharedExpertResult result = shared_expert.get();
                    layer_state.shared_expert_output = std::move(result.output);
                    record_captured_vulkan_execution_delta(statistics, result.vulkan);
                }
                continue;
            }
            if (node->type == ExecutionNodeType::Combine)
            {
                if (!layer_state.experts_executed)
                {
                    return Error{ErrorCode::InternalError, "Combine executed before its Expert wave"};
                }
                const auto combine_start = std::chrono::steady_clock::now();
                if (moe.has_shared_expert && layer_state.shared_expert_output.rows() == 0)
                {
                    ExpertExecutionMetrics shared_metrics;
                    layer_state.shared_expert_output = run_shared_expert(model, moe, layer_state.normalized, shared_metrics);
                }
                CpuBatch& moe_output = layer_state.normalized;
                moe_output.reset(hidden.rows(), model.descriptor.hidden_size, true);
                for (const ActiveExpertExecution& active : layer_state.active_experts)
                {
                    for (size_t batch_index = 0; batch_index < active.batch.routes.size(); ++batch_index)
                    {
                        const ExpertRoute& route = active.batch.routes[batch_index];
                        float* destination = moe_output.row(route.token_index);
                        const float* source = active.output.row(batch_index);
                        for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
                        {
                            destination[column] += route.weight * source[column];
                        }
                    }
                }
                if (moe.has_shared_expert)
                {
                    add_batch_inplace(moe_output, layer_state.shared_expert_output);
                }
                if (model.descriptor.hyper_connection_multiplier > 1)
                {
                    auto connected = hyper_connection_post(moe_output, hidden, layer_state.ffn_hyper_mix, model.descriptor.hyper_connection_multiplier);
                    if (!connected)
                        return connected.error();
                    hidden = std::move(connected).value();
                }
                else
                {
                    for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
                    {
                        float* hidden_row = hidden.row(token_index);
                        const float* output_row = moe_output.row(token_index);
                        for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
                            hidden_row[column] += output_row[column];
                    }
                }
                statistics.expert_combine_time_microseconds += elapsed_microseconds(combine_start);

                statistics.expert_time_microseconds += elapsed_microseconds(layer_state.expert_start);
                const auto target = std::find(
                    model.speculative.target_layer_ids.begin(),
                    model.speculative.target_layer_ids.end(),
                    layer.layer_id);
                if (target != model.speculative.target_layer_ids.end())
                {
                    capture_speculative_hidden(
                        hidden,
                        model.descriptor.hidden_size,
                        model.descriptor.hyper_connection_multiplier,
                        static_cast<size_t>(std::distance(model.speculative.target_layer_ids.begin(), target)),
                        state.speculative_main_hidden);
                }
                layer_state.reset();
                continue;
            }
            return Error{ErrorCode::InternalError, "unsupported execution graph node"};
        }
    }

    if (pending_router_prediction.result.valid())
    {
        return Error{
            ErrorCode::InternalError,
            "execution graph ended before a Router prediction target"};
    }

    if (model.expert_cache)
    {
        const ExpertCacheStatistics after = model.expert_cache->statistics();
        statistics.expert_cache_hits += after.hits - execution_cache_before.hits;
        statistics.expert_cache_misses += after.misses - execution_cache_before.misses;
        statistics.expert_cache_evictions += after.evictions - execution_cache_before.evictions;
        statistics.expert_cache_bytes_read += after.bytes_read - execution_cache_before.bytes_read;
        statistics.expert_cache_queued_reads += after.queued_reads - execution_cache_before.queued_reads;
        statistics.expert_cache_speculative_reads += after.speculative_reads - execution_cache_before.speculative_reads;
        statistics.expert_cache_cancelled_speculative_reads += after.cancelled_speculative_reads - execution_cache_before.cancelled_speculative_reads;
        statistics.expert_cache_dropped_speculative_admissions += after.dropped_speculative_admissions - execution_cache_before.dropped_speculative_admissions;
        statistics.expert_cache_unused_speculative_reads += after.unused_speculative_reads - execution_cache_before.unused_speculative_reads;
        statistics.expert_cache_short_term_reloads += after.short_term_reloads - execution_cache_before.short_term_reloads;
        statistics.expert_cache_arc_recent_bytes = after.arc_recent_bytes;
        statistics.expert_cache_arc_frequent_bytes = after.arc_frequent_bytes;
        statistics.expert_cache_arc_recent_target_bytes = after.arc_recent_target_bytes;
        statistics.expert_cache_arc_recent_ghost_bytes = after.arc_recent_ghost_bytes;
        statistics.expert_cache_arc_frequent_ghost_bytes = after.arc_frequent_ghost_bytes;
        statistics.expert_cache_arc_recent_ghost_hits += after.arc_recent_ghost_hits - execution_cache_before.arc_recent_ghost_hits;
        statistics.expert_cache_arc_frequent_ghost_hits += after.arc_frequent_ghost_hits - execution_cache_before.arc_frequent_ghost_hits;
        statistics.expert_cache_mapped_ranges += after.mapped_ranges - execution_cache_before.mapped_ranges;
        statistics.expert_cache_mapped_bytes += after.mapped_bytes - execution_cache_before.mapped_bytes;
        statistics.expert_cache_direct_read_ranges += after.direct_read_ranges - execution_cache_before.direct_read_ranges;
        statistics.expert_cache_direct_read_bytes += after.direct_read_bytes - execution_cache_before.direct_read_bytes;
        statistics.expert_cache_direct_read_fallbacks += after.direct_read_fallbacks - execution_cache_before.direct_read_fallbacks;
        statistics.expert_cache_buffered_read_ranges += after.buffered_read_ranges - execution_cache_before.buffered_read_ranges;
        statistics.expert_cache_buffered_read_bytes += after.buffered_read_bytes - execution_cache_before.buffered_read_bytes;
        statistics.expert_cache_coalesced_read_batches += after.coalesced_read_batches - execution_cache_before.coalesced_read_batches;
        statistics.expert_cache_coalesced_experts += after.coalesced_experts - execution_cache_before.coalesced_experts;
        statistics.expert_cache_coalesced_read_ranges_saved += after.coalesced_read_ranges_saved - execution_cache_before.coalesced_read_ranges_saved;
        statistics.expert_cache_read_policy = after.adaptive_read_policy;
        statistics.expert_cache_resident_bytes = after.resident_bytes;
        record_expert_victim_cache_delta(statistics, execution_cache_before.victim, after.victim);
    }
    if (model.expert_backend)
    {
        record_expert_backend_delta(statistics, expert_backend_before, model.expert_backend->statistics());
    }

    if (logits.empty())
        return Error{ErrorCode::InternalError, "execution graph did not produce logits"};
    record_vulkan_execution_delta(statistics, initial_vulkan_execution);
    auto residency = state.memory_manager.record_execution(model.graph);
    if (!residency)
        return residency.error();
    return logits;
}

Result<std::vector<std::vector<float>>> CpuExecutor::execute_decode_batch(const CompiledModel& model, std::span<const CpuDecodeBatchEntry> entries, CpuDecodeBatchMetrics& metrics) const
{
    const OpenMpHybridTeamLimit team_limit(model.hybrid_mode);
    if (entries.empty())
        return Error{ErrorCode::InvalidArgument, "decode batch cannot be empty"};

    const size_t session_count = entries.size();
    const uint32_t hyper_multiplier = model.descriptor.hyper_connection_multiplier;
    const uint32_t hyper_iterations = model.descriptor.hyper_connection_iterations;
    const float hyper_epsilon = model.descriptor.hyper_connection_epsilon;
    std::vector<CpuBatch> hidden(session_count);
    std::vector<std::vector<float>> logits(session_count);
    for (const CpuDecodeBatchEntry& entry : entries)
    {
        if (!entry.statistics || !entry.state)
        {
            return Error{ErrorCode::InvalidArgument, "decode batch entry is incomplete"};
        }
        if (entry.input_id < 0 || static_cast<uint32_t>(entry.input_id) >= model.descriptor.vocabulary_size)
        {
            return Error{ErrorCode::InvalidArgument, "token id is outside the model vocabulary"};
        }
        if (entry.statistics->expert_token_counts.size() < model.descriptor.expert_count)
        {
            entry.statistics->expert_token_counts.resize(model.descriptor.expert_count, 0);
        }
        if (entry.state->layers.size() != model.layers.size())
            entry.state->layers.resize(model.layers.size());
        if (entry.state->execution_layers.size() != model.layers.size())
        {
            entry.state->execution_layers.resize(model.layers.size());
        }
        for (LayerGraphState& layer_state : entry.state->execution_layers)
        {
            layer_state.reset();
        }
    }
    ExpertCacheStatistics cache_before;
    if (model.expert_cache)
        cache_before = model.expert_cache->statistics();
    ExpertBackendStatistics backend_before;
    if (model.expert_backend)
    {
        backend_before = model.expert_backend->statistics();
    }
    if (model.speculative.enabled())
    {
        const uint32_t speculative_hidden_columns =
            model.speculative.kind == SpeculativeModelKind::Mtp
                ? model.descriptor.hidden_size
                : model.descriptor.hidden_size
                      * static_cast<uint32_t>(
                          model.speculative.target_layer_ids.size());
        for (const CpuDecodeBatchEntry& entry : entries)
        {
            if (!entry.state->speculative_context_enabled)
                continue;
            entry.state->speculative_main_hidden.reset(1, speculative_hidden_columns, true);
            entry.state->speculative_main_hidden_position = entry.position_offset;
            if (entry.state->speculative_layers.size() != model.speculative.layers.size())
            {
                entry.state->speculative_layers.resize(model.speculative.layers.size());
            }
            if (model.speculative.kind == SpeculativeModelKind::Mtp)
            {
                entry.state->speculative_input_ids.assign(
                    1,
                    entry.input_id);
                entry.state->speculative_direct_alignment_ids.clear();
            }
        }
    }

    auto merge_rows_into = [session_count](const std::vector<CpuBatch>& batches, CpuBatch& merged) {
        if (batches.empty() || batches.front().rows() != 1)
            return false;
        merged.reset(session_count, batches.front().columns(), false);
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            if (batches[session_index].rows() != 1 || batches[session_index].columns() != merged.columns())
            {
                return false;
            }
            std::copy_n(batches[session_index].row(0), merged.columns(), merged.row(session_index));
        }
        return true;
    };
    auto split_rows = [session_count](const CpuBatch& merged, std::vector<CpuBatch>& batches) {
        batches.resize(session_count);
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            batches[session_index].reset(1, merged.columns(), false);
            std::copy_n(merged.row(session_index), merged.columns(), batches[session_index].row(0));
        }
    };
    auto split_hyper_mix = [session_count, hyper_multiplier](const CpuHyperConnectionMix& merged, std::vector<CpuHyperConnectionMix>& mixes) {
        mixes.resize(session_count);
        const size_t post_stride = hyper_multiplier;
        const size_t combine_stride = static_cast<size_t>(hyper_multiplier) * hyper_multiplier;
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            mixes[session_index].reduced.reset(1, merged.reduced.columns(), false);
            std::copy_n(merged.reduced.row(session_index), merged.reduced.columns(), mixes[session_index].reduced.row(0));
            const auto post_begin = merged.post.begin() + session_index * post_stride;
            mixes[session_index].post.assign(post_begin, post_begin + post_stride);
            const auto combine_begin = merged.combine.begin() + session_index * combine_stride;
            mixes[session_index].combine.assign(combine_begin, combine_begin + combine_stride);
        }
    };
    auto merge_hyper_mixes = [session_count, hyper_multiplier](const std::vector<CpuHyperConnectionMix>& mixes, CpuHyperConnectionMix& merged) {
        const size_t post_stride = hyper_multiplier;
        const size_t combine_stride = static_cast<size_t>(hyper_multiplier) * hyper_multiplier;
        merged.post.resize(session_count * post_stride);
        merged.combine.resize(session_count * combine_stride);
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            std::copy_n(mixes[session_index].post.data(), post_stride, merged.post.data() + session_index * post_stride);
            std::copy_n(mixes[session_index].combine.data(), combine_stride, merged.combine.data() + session_index * combine_stride);
        }
    };

    ExpertDispatcher dispatcher;
    for (const ExecutionWave& wave : model.schedule.waves)
    {
        for (ExecutionNodeId node_id : wave.nodes)
        {
            const ExecutionNode* node = model.graph.find(node_id);
            if (!node)
            {
                return Error{ErrorCode::InternalError, "execution schedule references an invalid node"};
            }

            if (node->type == ExecutionNodeType::TokenEmbedding)
            {
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                std::vector<int32_t>& input_ids = scratch.staged_input_ids;
                input_ids.resize(session_count);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    input_ids[session_index] = entries[session_index].input_id;
                }
                embedding_batch_into(model.weights.at(model.token_embedding), input_ids, scratch.staged_output);
                if (hyper_multiplier > 1)
                {
                    expand_hyper_connections(scratch.staged_output, hyper_multiplier, scratch.staged_merged);
                }
                split_rows(scratch.staged_output, hidden);
                const uint64_t elapsed = elapsed_microseconds(start);
                for (const CpuDecodeBatchEntry& entry : entries)
                    entry.statistics->embedding_time_microseconds += elapsed;
                continue;
            }
            if (node->type == ExecutionNodeType::FinalNorm)
            {
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                if (hyper_multiplier > 1)
                {
                    CpuBatch& merged_hyper = scratch.staged_merged;
                    if (!merge_rows_into(hidden, merged_hyper))
                    {
                        return Error{
                            ErrorCode::InternalError,
                            "cannot merge staged hyper head rows"};
                    }
                    auto head = hyper_connection_head(merged_hyper, model.weights.at(model.hyper_head_function), model.weights.at(model.hyper_head_scale), model.weights.at(model.hyper_head_base),
                                                      hyper_multiplier, model.descriptor.norm_epsilon, hyper_epsilon);
                    if (!head)
                        return head.error();
                    split_rows(head.value(), hidden);
                }
                CpuBatch& merged = scratch.staged_merged;
                if (!merge_rows_into(hidden, merged))
                {
                    return Error{ErrorCode::InternalError, "cannot merge staged hidden rows"};
                }
                rms_norm_batch_into(merged, model.weights.at(model.final_norm_weight), model.descriptor.norm_epsilon, scratch.staged_output, model.descriptor.norm_weight_offset);
                split_rows(scratch.staged_output, hidden);
                if (model.speculative.kind == SpeculativeModelKind::Mtp)
                {
                    for (size_t session_index = 0;
                         session_index < session_count;
                         ++session_index)
                    {
                        CpuSessionState& state =
                            *entries[session_index].state;
                        if (!state.speculative_context_enabled)
                            continue;
                        std::copy_n(
                            hidden[session_index].row(0),
                            model.descriptor.hidden_size,
                            state.speculative_main_hidden.row(0));
                    }
                }
                const uint64_t elapsed = elapsed_microseconds(start);
                for (const CpuDecodeBatchEntry& entry : entries)
                    entry.statistics->final_norm_time_microseconds += elapsed;
                continue;
            }
            if (node->type == ExecutionNodeType::LmHead)
            {
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                CpuBatch& merged = scratch.staged_merged;
                if (!merge_rows_into(hidden, merged))
                {
                    return Error{ErrorCode::InternalError, "cannot merge staged LM head rows"};
                }
                const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution();
                linear_batch_into(model.weights.at(model.lm_head_weight), merged, scratch.staged_output);
                const std::vector<std::vector<float>> merged_logits = batch_to_vectors(scratch.staged_output);
                const uint64_t elapsed = elapsed_microseconds(start);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    logits[session_index] = merged_logits[session_index];
                    entries[session_index].statistics->lm_head_time_microseconds += elapsed;
                }
                for (const CpuDecodeBatchEntry& entry : entries)
                    record_vulkan_execution_delta(*entry.statistics, vulkan_before);
                continue;
            }
            if (node->layer_id >= model.layers.size())
            {
                return Error{ErrorCode::InternalError, "execution node layer is out of range"};
            }

            const CompiledLayerPlan& layer = model.layers[node->layer_id];
            const MoeBlockPlan& moe = layer.moe;
            if (node->type == ExecutionNodeType::Attention)
            {
                if (has_flag(layer.attention.flags, AttentionBlockGatedDeltaNet))
                {
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        const auto start = std::chrono::steady_clock::now();
                        const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution();
                        execute_gated_delta_net_into(
                            model.weights,
                            layer.attention,
                            model.descriptor.norm_epsilon,
                            state.layers[layer.layer_id],
                            state.gated_delta_scratch,
                            hidden[session_index],
                            state.gated_delta_scratch.output);
                        hidden[session_index].swap(state.gated_delta_scratch.output);
                        SessionStatistics& statistics = *entries[session_index].statistics;
                        statistics.attention_time_microseconds += elapsed_microseconds(start);
                        record_vulkan_execution_delta(statistics, vulkan_before);
                    }
                }
                else if (has_flag(layer.attention.flags, AttentionBlockLatent))
                {
                    const auto start = std::chrono::steady_clock::now();
                    const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution();
                    std::vector<uint64_t> positions(session_count);
                    std::vector<CpuLayerCache*> caches(session_count);
                    CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                    CpuBatch& merged_hidden = scratch.staged_merged;
                    if (!merge_rows_into(hidden, merged_hidden))
                    {
                        return Error{
                            ErrorCode::InternalError,
                            "cannot merge staged attention rows"};
                    }
                    CpuHyperConnectionMix merged_mix;
                    const CpuBatch* attention_input = &merged_hidden;
                    if (hyper_multiplier > 1)
                    {
                        auto mixed = hyper_connection_pre(merged_hidden, model.weights.at(layer.hyper_connection.attention_function), model.weights.at(layer.hyper_connection.attention_scale),
                                                          model.weights.at(layer.hyper_connection.attention_base), hyper_multiplier, hyper_iterations, model.descriptor.norm_epsilon, hyper_epsilon);
                        if (!mixed)
                            return mixed.error();
                        merged_mix = std::move(mixed).value();
                        attention_input = &merged_mix.reduced;
                    }
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        positions[session_index] = entries[session_index].position_offset;
                        caches[session_index] = &state.layers[layer.layer_id];
                    }
                    auto merged_output = execute_latent_attention_batch(model.weights, layer.attention, model.descriptor.norm_epsilon, positions, caches, *attention_input);
                    if (!merged_output)
                        return merged_output.error();
                    if (hyper_multiplier > 1)
                    {
                        auto connected = hyper_connection_post(merged_output.value(), merged_hidden, merged_mix, hyper_multiplier);
                        if (!connected)
                            return connected.error();
                        split_rows(connected.value(), hidden);
                    }
                    else
                    {
                        std::vector<CpuBatch> attention_outputs;
                        split_rows(merged_output.value(), attention_outputs);
                        for (size_t session_index = 0; session_index < session_count; ++session_index)
                        {
                            add_batch_inplace(hidden[session_index], attention_outputs[session_index]);
                        }
                    }
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        SessionStatistics& statistics = *entries[session_index].statistics;
                        statistics.attention_time_microseconds += elapsed_microseconds(start);
                        record_vulkan_execution_delta(*entries[session_index].statistics, vulkan_before);
                    }
                }
                else
                {
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        const auto start = std::chrono::steady_clock::now();
                        const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution();
                        execute_attention_block_into(model.weights, layer.attention, model.descriptor.norm_epsilon, model.descriptor.kv_cache_dtype, entries[session_index].position_offset,
                                                     state.layers[layer.layer_id], state.attention_scratch, hidden[session_index], state.attention_scratch.output);
                        hidden[session_index].swap(state.attention_scratch.output);
                        SessionStatistics& statistics = *entries[session_index].statistics;
                        statistics.attention_time_microseconds += elapsed_microseconds(start);
                        record_vulkan_execution_delta(*entries[session_index].statistics, vulkan_before);
                    }
                }
                continue;
            }
            if (node->type == ExecutionNodeType::Router)
            {
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                CpuBatch& merged_hidden = scratch.staged_merged;
                CpuBatch merged_hyper;
                if (!merge_rows_into(hidden, merged_hyper))
                {
                    return Error{
                        ErrorCode::InternalError,
                        "cannot merge staged FFN hyper rows"};
                }
                if (hyper_multiplier > 1)
                {
                    auto mixed = hyper_connection_pre(merged_hyper, model.weights.at(layer.hyper_connection.ffn_function), model.weights.at(layer.hyper_connection.ffn_scale),
                                                      model.weights.at(layer.hyper_connection.ffn_base), hyper_multiplier, hyper_iterations, model.descriptor.norm_epsilon, hyper_epsilon);
                    if (!mixed)
                        return mixed.error();
                    CpuHyperConnectionMix merged_mix = std::move(mixed).value();
                    rms_norm_batch_into(merged_mix.reduced, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, merged_hidden, model.descriptor.norm_weight_offset);
                    std::vector<CpuHyperConnectionMix> mixes;
                    split_hyper_mix(merged_mix, mixes);
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                        layer_state.ffn_hyper_mix = std::move(mixes[session_index]);
                    }
                }
                else
                {
                    rms_norm_batch_into(merged_hyper, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, merged_hidden, model.descriptor.norm_weight_offset);
                }
                std::vector<CpuBatch> normalized;
                split_rows(merged_hidden, normalized);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    layer_state.normalized = std::move(normalized[session_index]);
                }
                CpuBatch& merged_logits = scratch.staged_router_logits;
                linear_batch_into(model.weights.at(moe.router_weight), merged_hidden, merged_logits);
                if (moe.router_bias != invalid_tensor_handle)
                {
                    add_bias_inplace(merged_logits, model.weights.at(moe.router_bias));
                }
                const auto router_start = std::chrono::steady_clock::now();
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    layer_state.router_logits.reset(1, merged_logits.columns(), false);
                    std::copy_n(merged_logits.row(session_index), merged_logits.columns(), layer_state.router_logits.row(0));
                    layer_state.router_start = router_start;
                }
                const uint64_t elapsed = elapsed_microseconds(start);
                for (const CpuDecodeBatchEntry& entry : entries)
                    entry.statistics->router_time_microseconds += elapsed;
                continue;
            }
            if (node->type == ExecutionNodeType::ExpertDispatch)
            {
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    CpuSessionState& state = *entries[session_index].state;
                    LayerGraphState& layer_state = state.execution_layers[layer.layer_id];
                    SessionStatistics& statistics = *entries[session_index].statistics;
                    ExpertDispatchOptions options;
                    options.expert_count = static_cast<uint32_t>(moe.experts.size());
                    options.top_k = moe.top_k;
                    options.score_function = moe.score_function;
                    options.normalization = moe.normalization;
                    options.routed_scaling_factor = moe.routed_scaling_factor;
                    if (moe.router_selection_bias != invalid_tensor_handle)
                    {
                        const TensorData& selection_bias = model.weights.at(moe.router_selection_bias);
                        options.selection_bias = selection_bias.float32_values();
                    }
                    std::vector<uint32_t> explicit_expert_ids;
                    if (moe.token_experts != invalid_tensor_handle)
                    {
                        const std::span<const int64_t> table = model.weights.at(moe.token_experts).int64_values();
                        explicit_expert_ids.resize(moe.top_k);
                        for (uint32_t route = 0; route < moe.top_k; ++route)
                        {
                            explicit_expert_ids[route] = static_cast<uint32_t>(table[static_cast<size_t>(entries[session_index].input_id) * moe.top_k + route]);
                        }
                        options.explicit_expert_ids = explicit_expert_ids;
                    }
                    options.flags = has_flag(moe.flags, MoeBlockNormalizeTopKWeights) ? static_cast<uint32_t>(ExpertDispatchNormalizeTopKWeights) : 0u;
                    auto dispatched = dispatcher.dispatch_into(layer_state.router_logits.values(), 1, options, layer_state.dispatch_plan);
                    if (!dispatched)
                        return dispatched.error();
                    const ExpertDispatchPlan& plan = layer_state.dispatch_plan;
                    statistics.expert_assignments += plan.assignment_count;
                    layer_state.active_experts.resize(plan.batches.size());
                    resolve_router_predictions(model, layer, plan, state, statistics, false);
                    for (size_t batch_index = 0; batch_index < plan.batches.size(); ++batch_index)
                    {
                        const ExpertBatch& batch = plan.batches[batch_index];
                        statistics.expert_token_counts[batch.expert_id] += batch.routes.size();
                        const ExpertPlan& expert = moe.experts[batch.expert_id];
                        record_expert_weight_demand(expert, batch.routes.size(), statistics);
                        if (expert.runtime)
                        {
                            expert.runtime->record_dispatch(batch.routes.size(), statistics.expert_assignments);
                        }
                        ActiveExpertExecution& active = layer_state.active_experts[batch_index];
                        active.prepare(batch);
                    }
                    layer_state.router_logits.clear();
                    layer_state.expert_start = std::chrono::steady_clock::now();
                }
                continue;
            }
            if (node->type == ExecutionNodeType::Expert || node->type == ExecutionNodeType::ExpertGroup)
            {
                struct RouteOrigin
                {
                    size_t session_index = 0;
                    size_t active_index = 0;
                    size_t route_index = 0;
                };
                LayerGraphState combined;
                combined.normalized.reset(session_count, model.descriptor.hidden_size, false);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    const LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    std::copy_n(layer_state.normalized.row(0), combined.normalized.columns(), combined.normalized.row(session_index));
                }
                const size_t missing = std::numeric_limits<size_t>::max();
                std::vector<size_t> combined_by_expert(moe.experts.size(), missing);
                std::vector<std::vector<RouteOrigin>> origins;
                uint64_t logical_batches = 0;
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    logical_batches += layer_state.active_experts.size();
                    for (size_t active_index = 0; active_index < layer_state.active_experts.size(); ++active_index)
                    {
                        ActiveExpertExecution& source = layer_state.active_experts[active_index];
                        const uint32_t expert_id = source.batch.expert_id;
                        size_t combined_index = combined_by_expert[expert_id];
                        if (combined_index == missing)
                        {
                            combined_index = combined.active_experts.size();
                            combined_by_expert[expert_id] = combined_index;
                            ExpertBatch batch;
                            batch.expert_id = expert_id;
                            combined.active_experts.emplace_back();
                            combined.active_experts.back().prepare(std::move(batch));
                            origins.emplace_back();
                        }
                        ActiveExpertExecution& destination = combined.active_experts[combined_index];
                        for (size_t route_index = 0; route_index < source.batch.routes.size(); ++route_index)
                        {
                            ExpertRoute route = source.batch.routes[route_index];
                            route.token_index = static_cast<uint32_t>(session_index);
                            destination.batch.routes.push_back(route);
                            origins[combined_index].push_back({session_index, active_index, route_index});
                        }
                    }
                }

                SessionStatistics aggregate_statistics;
                aggregate_statistics.expert_token_counts.resize(model.descriptor.expert_count, 0);
                std::future<OverlappedSharedExpertResult> shared_expert;
                if (can_overlap_vulkan_shared_expert(model, moe))
                {
                    CpuBatch shared_input = combined.normalized;
                    shared_expert = std::async(std::launch::async, [&model, &moe, shared_input = std::move(shared_input)]() mutable {
                        return run_overlapped_shared_expert(model, moe, shared_input);
                    });
                }
                const auto engine_start = std::chrono::steady_clock::now();
                auto executed = run_moe(model, moe, combined, aggregate_statistics, entries.front().state->expert_scratch, layer.layer_id);
                const uint64_t engine_elapsed = elapsed_microseconds(engine_start);
                if (!executed)
                    return executed.error();
                if (shared_expert.valid())
                {
                    OverlappedSharedExpertResult result = shared_expert.get();
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                        layer_state.shared_expert_output.reset(1, result.output.columns(), false);
                        std::copy_n(result.output.row(session_index), result.output.columns(), layer_state.shared_expert_output.row(0));
                        record_captured_vulkan_execution_delta(*entries[session_index].statistics, result.vulkan);
                    }
                }

                metrics.logical_expert_batches += logical_batches;
                metrics.physical_expert_batches += combined.active_experts.size();
                for (const ActiveExpertExecution& active : combined.active_experts)
                {
                    metrics.max_expert_batch_size = std::max<uint64_t>(metrics.max_expert_batch_size, active.batch.routes.size());
                    if (active.batch.routes.size() > 1)
                    {
                        metrics.coalesced_expert_routes += active.batch.routes.size();
                    }
                }
                for (size_t combined_index = 0; combined_index < combined.active_experts.size(); ++combined_index)
                {
                    const ActiveExpertExecution& active = combined.active_experts[combined_index];
                    for (size_t route_index = 0; route_index < origins[combined_index].size(); ++route_index)
                    {
                        const ExpertRoute& route = active.batch.routes[route_index];
                        if (route.rank >= maximum_expert_route_ranks)
                            continue;
                        SessionStatistics& route_statistics = *entries[origins[combined_index][route_index].session_index].statistics;
                        ++route_statistics.expert_route_rank_demands[route.rank];
                        route_statistics.expert_route_rank_demand_queue_time_microseconds[route.rank] += active.metrics.cache_wait_time_microseconds;
                    }
                }

                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    SessionStatistics& statistics = *entries[session_index].statistics;
                    statistics.expert_engine_time_microseconds += engine_elapsed;
                    statistics.expert_compute_time_microseconds += aggregate_statistics.expert_compute_time_microseconds;
                    statistics.expert_cache_wait_time_microseconds += aggregate_statistics.expert_cache_wait_time_microseconds;
                    statistics.expert_cache_management_time_microseconds += aggregate_statistics.expert_cache_management_time_microseconds;
                    statistics.expert_regroup_time_microseconds += aggregate_statistics.expert_regroup_time_microseconds;
                    if (session_index == 0)
                    {
                        statistics.mxfp4_reused_input_rows += aggregate_statistics.mxfp4_reused_input_rows;
                    }
                    statistics.expert_batches += layer_state.active_experts.size();
                    for (ActiveExpertExecution& active : layer_state.active_experts)
                    {
                        active.output.reset(active.batch.routes.size(), model.descriptor.hidden_size, false);
                        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                        ExpertExecutionMetrics logical_metrics;
                        if (expert.gate_up_weight != invalid_tensor_handle)
                        {
                            const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                            const TensorData& down = model.weights.at(expert.down_weight);
                            record_mxfp4(gate_up, active.batch.routes.size(), logical_metrics);
                            record_mxfp4(down, active.batch.routes.size(), logical_metrics);
                            if (gate_up.dtype == DType::MxFp4)
                            {
                                logical_metrics.mxfp4_fused_gate_up_rows += static_cast<uint64_t>(active.batch.routes.size()) * gate_up.shape[0] / 2;
                            }
                        }
                        statistics.mxfp4_decode_gemv_rows += logical_metrics.mxfp4_decode_gemv_rows;
                        statistics.mxfp4_prefill_gemm_rows += logical_metrics.mxfp4_prefill_gemm_rows;
                        statistics.mxfp4_paired_rows += logical_metrics.mxfp4_paired_rows;
                        statistics.mxfp4_fused_gate_up_rows += logical_metrics.mxfp4_fused_gate_up_rows;
                    }
                    layer_state.experts_executed = true;
                }
                for (size_t combined_index = 0; combined_index < combined.active_experts.size(); ++combined_index)
                {
                    const ActiveExpertExecution& source = combined.active_experts[combined_index];
                    for (size_t route_index = 0; route_index < origins[combined_index].size(); ++route_index)
                    {
                        const RouteOrigin& origin = origins[combined_index][route_index];
                        ActiveExpertExecution& destination = entries[origin.session_index].state->execution_layers[layer.layer_id].active_experts[origin.active_index];
                        std::copy_n(source.output.row(route_index), model.descriptor.hidden_size, destination.output.row(origin.route_index));
                    }
                }
                continue;
            }
            if (node->type == ExecutionNodeType::Combine)
            {
                const auto combine_start = std::chrono::steady_clock::now();
                std::vector<CpuBatch> moe_outputs(session_count);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    if (!layer_state.experts_executed)
                    {
                        return Error{ErrorCode::InternalError, "Combine executed before its Expert wave"};
                    }
                    if (moe.has_shared_expert && layer_state.shared_expert_output.rows() == 0)
                    {
                        ExpertExecutionMetrics shared_metrics;
                        layer_state.shared_expert_output = run_shared_expert(model, moe, layer_state.normalized, shared_metrics);
                    }
                    CpuBatch& moe_output = layer_state.normalized;
                    moe_output.reset(1, model.descriptor.hidden_size, true);
                    for (const ActiveExpertExecution& active : layer_state.active_experts)
                    {
                        for (size_t batch_index = 0; batch_index < active.batch.routes.size(); ++batch_index)
                        {
                            const ExpertRoute& route = active.batch.routes[batch_index];
                            float* destination = moe_output.row(route.token_index);
                            const float* source = active.output.row(batch_index);
                            for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
                            {
                                destination[column] += route.weight * source[column];
                            }
                        }
                    }
                    if (moe.has_shared_expert)
                    {
                        add_batch_inplace(moe_output, layer_state.shared_expert_output);
                    }
                    moe_outputs[session_index] = std::move(moe_output);
                }
                if (hyper_multiplier > 1)
                {
                    CpuBatch merged_branch;
                    CpuBatch merged_residual;
                    CpuHyperConnectionMix merged_mix;
                    std::vector<CpuHyperConnectionMix> mixes(session_count);
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                        mixes[session_index] = std::move(layer_state.ffn_hyper_mix);
                    }
                    merge_hyper_mixes(mixes, merged_mix);
                    if (!merge_rows_into(moe_outputs, merged_branch) || !merge_rows_into(hidden, merged_residual))
                    {
                        return Error{
                            ErrorCode::InternalError,
                            "cannot merge staged FFN post rows"};
                    }
                    auto connected = hyper_connection_post(merged_branch, merged_residual, merged_mix, hyper_multiplier);
                    if (!connected)
                        return connected.error();
                    split_rows(connected.value(), hidden);
                }
                else
                {
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        add_batch_inplace(hidden[session_index], moe_outputs[session_index]);
                    }
                }
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    const auto target = std::find(model.speculative.target_layer_ids.begin(), model.speculative.target_layer_ids.end(), layer.layer_id);
                    if (target != model.speculative.target_layer_ids.end())
                    {
                        capture_speculative_hidden(hidden[session_index], model.descriptor.hidden_size, hyper_multiplier,
                                                   static_cast<size_t>(std::distance(model.speculative.target_layer_ids.begin(), target)),
                                                   entries[session_index].state->speculative_main_hidden);
                    }
                    SessionStatistics& statistics = *entries[session_index].statistics;
                    statistics.expert_combine_time_microseconds += elapsed_microseconds(combine_start);
                    statistics.expert_time_microseconds += elapsed_microseconds(layer_state.expert_start);
                    layer_state.reset();
                }
                continue;
            }
            return Error{ErrorCode::InternalError, "unsupported execution graph node"};
        }
    }

    if (model.expert_cache)
    {
        const ExpertCacheStatistics cache_after = model.expert_cache->statistics();
        for (const CpuDecodeBatchEntry& entry : entries)
        {
            SessionStatistics& statistics = *entry.statistics;
            statistics.expert_cache_hits += cache_after.hits - cache_before.hits;
            statistics.expert_cache_misses += cache_after.misses - cache_before.misses;
            statistics.expert_cache_evictions += cache_after.evictions - cache_before.evictions;
            statistics.expert_cache_bytes_read += cache_after.bytes_read - cache_before.bytes_read;
            statistics.expert_cache_speculative_reads += cache_after.speculative_reads - cache_before.speculative_reads;
            statistics.expert_cache_cancelled_speculative_reads += cache_after.cancelled_speculative_reads - cache_before.cancelled_speculative_reads;
            statistics.expert_cache_dropped_speculative_admissions += cache_after.dropped_speculative_admissions - cache_before.dropped_speculative_admissions;
            statistics.expert_cache_unused_speculative_reads += cache_after.unused_speculative_reads - cache_before.unused_speculative_reads;
            statistics.expert_cache_short_term_reloads += cache_after.short_term_reloads - cache_before.short_term_reloads;
            statistics.expert_cache_coalesced_read_batches += cache_after.coalesced_read_batches - cache_before.coalesced_read_batches;
            statistics.expert_cache_coalesced_experts += cache_after.coalesced_experts - cache_before.coalesced_experts;
            statistics.expert_cache_coalesced_read_ranges_saved += cache_after.coalesced_read_ranges_saved - cache_before.coalesced_read_ranges_saved;
            statistics.expert_cache_resident_bytes = cache_after.resident_bytes;
            statistics.expert_cache_arc_recent_bytes = cache_after.arc_recent_bytes;
            statistics.expert_cache_arc_frequent_bytes = cache_after.arc_frequent_bytes;
            statistics.expert_cache_arc_recent_target_bytes = cache_after.arc_recent_target_bytes;
            record_expert_victim_cache_delta(statistics, cache_before.victim, cache_after.victim);
        }
    }
    if (model.expert_backend)
    {
        const ExpertBackendStatistics backend_after = model.expert_backend->statistics();
        for (const CpuDecodeBatchEntry& entry : entries)
        {
            record_expert_backend_delta(*entry.statistics, backend_before, backend_after);
        }
    }
    for (size_t session_index = 0; session_index < session_count; ++session_index)
    {
        if (logits[session_index].empty())
        {
            return Error{ErrorCode::InternalError, "staged execution graph did not produce logits"};
        }
        auto residency = entries[session_index].state->memory_manager.record_execution(model.graph);
        if (!residency)
            return residency.error();
    }
    return logits;
}

} // namespace moe
} // namespace ncnn
