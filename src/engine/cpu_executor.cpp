#include "cpu_executor.h"

#include "kernels/cpu_attention.h"
#include "kernels/cpu_batch.h"
#include "kernels/cpu_ops.h"
#include "cpu_session_state.h"
#include "expert_backend.h"
#include "storage/expert_cache.h"
#include "backends/ncnn/ncnn_attention.h"
#include "backends/ncnn/ncnn_linear.h"

#include "ncnn/moe/expert_dispatcher.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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

static uint64_t elapsed_microseconds(std::chrono::steady_clock::time_point start)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

static bool read_ready_first_option() noexcept
{
#if defined(_MSC_VER)
    char* value = nullptr;
    size_t value_size = 0;
    if (_dupenv_s(&value, &value_size, "NCNN_MOE_EXPERT_READY_FIRST") != 0 || !value)
    {
        return true;
    }
    const bool result = std::string_view(value) != "0";
    std::free(value);
    return result;
#else
    const char* value = std::getenv("NCNN_MOE_EXPERT_READY_FIRST");
    return !value || std::string_view(value) != "0";
#endif
}

static bool wait_for_any_ready_expert() noexcept
{
    static const bool enabled = read_ready_first_option();
    return enabled;
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

static void record_vulkan_execution_delta(SessionStatistics& statistics, const VulkanExecutionSnapshot& before)
{
    const uint64_t dispatches = NcnnLinearOperator::current_thread_vulkan_dispatches();
    const uint64_t attention_blocks = NcnnVulkanAttentionOperator::current_thread_blocks();
    const NcnnVulkanRuntimeCounters after = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
    statistics.vulkan_linear_dispatches += dispatches - before.dispatches;
    statistics.vulkan_attention_blocks += attention_blocks - before.attention_blocks;
    statistics.vulkan_compute_submissions += after.compute_submissions - before.counters.compute_submissions;
    statistics.vulkan_async_submissions += after.asynchronous_submissions - before.counters.asynchronous_submissions;
    statistics.vulkan_batch_uploads += after.batch_uploads - before.counters.batch_uploads;
    statistics.vulkan_batch_downloads += after.batch_downloads - before.counters.batch_downloads;
    statistics.vulkan_auxiliary_uploads += after.auxiliary_uploads - before.counters.auxiliary_uploads;
    statistics.vulkan_auxiliary_upload_bytes += after.auxiliary_upload_bytes - before.counters.auxiliary_upload_bytes;
    statistics.vulkan_staging_slot_resizes += after.staging_slot_resizes - before.counters.staging_slot_resizes;
    statistics.vulkan_staging_slot_reuses += after.staging_slot_reuses - before.counters.staging_slot_reuses;
    statistics.vulkan_staging_slot_acquisitions += after.staging_slot_acquisitions - before.counters.staging_slot_acquisitions;
    statistics.vulkan_staging_slot_contentions += after.staging_slot_contentions - before.counters.staging_slot_contentions;
    statistics.vulkan_command_buffer_reuses += after.command_buffer_reuses - before.counters.command_buffer_reuses;
    statistics.vulkan_attention_qkv_rope_fusions += after.attention_qkv_rope_fusions - before.counters.attention_qkv_rope_fusions;
    statistics.vulkan_attention_qkv_ring_fusions += after.attention_qkv_ring_fusions - before.counters.attention_qkv_ring_fusions;
    statistics.vulkan_attention_decode_sdpa_fusions += after.attention_decode_sdpa_fusions - before.counters.attention_decode_sdpa_fusions;
    statistics.vulkan_kv_ring_appends += after.kv_ring_appends - before.counters.kv_ring_appends;
    statistics.vulkan_kv_ring_resizes += after.kv_ring_resizes - before.counters.kv_ring_resizes;
    statistics.vulkan_kv_ring_wrapped_views += after.kv_ring_wrapped_views - before.counters.kv_ring_wrapped_views;
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
            activated = fused_mxfp4_gate_up_batch(gate_up_weight, gate_up_bias, input, expert.activation_limit);
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
    CpuBatch up = expert_linear(weights.at(expert.up_weight), nullptr, input, metrics);
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
                up_row[column] *= activate(gate_row[column], expert.activation, expert.activation_limit);
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
            record_mxfp4(*decode_tasks[task_index].down, 1, active.metrics);
            active.metrics.mxfp4_fused_gate_up_rows += decode_tasks[task_index].gate_up->shape[0] / 2;
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
    return expert.activation == ExpertActivation::GptOssSwiGlu
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
                                expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias), residency_group, token_count, expert.activation_limit);
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
    metadata.enabled = true;
    return metadata;
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
            if (expert.gate_up_weight == invalid_tensor_handle || expert.activation != ExpertActivation::GptOssSwiGlu)
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
        auto acquired = model.expert_cache->wait_acquire_ready_pairs(requests, leases, wait_for_any_ready_expert());
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
        ++statistics.expert_batches;
    }
    layer_state.experts_executed = true;
    return {};
}

Result<std::vector<std::vector<float>>> CpuExecutor::execute(const CompiledModel& model, std::span<const int32_t> input_ids, SessionStatistics& statistics, CpuSessionState& state, uint64_t position_offset) const
{
    const uint64_t initial_vulkan_dispatches = NcnnLinearOperator::current_thread_vulkan_dispatches();
    const uint64_t initial_vulkan_attention_blocks = NcnnVulkanAttentionOperator::current_thread_blocks();
    const NcnnVulkanRuntimeCounters initial_vulkan_runtime_counters = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
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
    std::vector<std::vector<float>> logits;
    std::vector<LayerGraphState>& layer_states = state.execution_layers;
    ExpertDispatcher dispatcher;
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
                statistics.embedding_time_microseconds += elapsed_microseconds(embedding_start);
                continue;
            }
            if (node->type == ExecutionNodeType::FinalNorm)
            {
                const auto final_norm_start = std::chrono::steady_clock::now();
                rms_norm_batch_into(hidden, model.weights.at(model.final_norm_weight), model.descriptor.norm_epsilon, state.expert_scratch.staged_output);
                hidden.swap(state.expert_scratch.staged_output);
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
                if (model.expert_cache && hidden.rows() == 1 && layer.layer_id < state.layers.size())
                {
                    for (uint32_t expert_id : state.layers[layer.layer_id].previous_expert_ids)
                    {
                        if (expert_id >= moe.experts.size())
                            continue;
                        const ExpertPlan& previous = moe.experts[expert_id];
                        if (previous.gate_up_weight == invalid_tensor_handle)
                        {
                            continue;
                        }
                        const TensorData& gate_up = model.weights.at(previous.gate_up_weight);
                        const TensorData& down = model.weights.at(previous.down_weight);
                        if (gate_up.mxfp4_file_storage || down.mxfp4_file_storage)
                        {
                            ++statistics.expert_route_predictions;
                            const auto prediction = model.expert_cache->prefetch_pair(gate_up, down, layer.layer_id, previous.cache_key);
                            const bool prediction_ready = prediction && prediction.value();
                            if (prediction_ready)
                            {
                                ++statistics.expert_route_prediction_cache_hits;
                            }
                            else
                            {
                                ++statistics.expert_route_prediction_cache_misses;
                            }
                        }
                    }
                }
                const auto attention_start = std::chrono::steady_clock::now();
                execute_attention_block_into(model.weights, layer.attention, model.descriptor.norm_epsilon, model.descriptor.kv_cache_dtype, position_offset, state.layers[layer.layer_id], state.attention_scratch, hidden,
                                             state.attention_scratch.output);
                hidden.swap(state.attention_scratch.output);
                statistics.attention_time_microseconds += elapsed_microseconds(attention_start);
                continue;
            }
            if (node->type == ExecutionNodeType::Router)
            {
                if (model.expert_cache)
                    model.expert_cache->cancel_prediction();
                layer_state.router_start = std::chrono::steady_clock::now();
                rms_norm_batch_into(hidden, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized);
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
                options.normalization = moe.normalization;
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
                {
                    const std::vector<uint32_t>& previous_expert_ids = state.layers[layer.layer_id].previous_expert_ids;
                    for (const ExpertBatch& batch : plan.batches)
                    {
                        if (std::find(previous_expert_ids.begin(), previous_expert_ids.end(), batch.expert_id) != previous_expert_ids.end())
                        {
                            ++statistics.expert_route_prediction_matches;
                        }
                    }
                    state.layers[layer.layer_id].previous_expert_ids.clear();
                    state.layers[layer.layer_id].previous_expert_ids.reserve(plan.batches.size());
                }
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
                    if (hidden.rows() == 1 && layer.layer_id < state.layers.size())
                    {
                        state.layers[layer.layer_id].previous_expert_ids.push_back(active.batch.expert_id);
                    }
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
                const auto expert_engine_start = std::chrono::steady_clock::now();
                auto executed = run_moe(model, moe, layer_state, statistics, state.expert_scratch, layer.layer_id);
                statistics.expert_engine_time_microseconds += elapsed_microseconds(expert_engine_start);
                if (!executed)
                    return executed.error();
                continue;
            }
            if (node->type == ExecutionNodeType::Combine)
            {
                if (!layer_state.experts_executed)
                {
                    return Error{ErrorCode::InternalError, "Combine executed before its Expert wave"};
                }
                const auto combine_start = std::chrono::steady_clock::now();
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
                for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
                {
                    float* hidden_row = hidden.row(token_index);
                    const float* output_row = moe_output.row(token_index);
                    for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
                    {
                        hidden_row[column] += output_row[column];
                    }
                }
                statistics.expert_combine_time_microseconds += elapsed_microseconds(combine_start);

                statistics.expert_time_microseconds += elapsed_microseconds(layer_state.expert_start);
                layer_state.reset();
                continue;
            }
            return Error{ErrorCode::InternalError, "unsupported execution graph node"};
        }
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
    statistics.vulkan_linear_dispatches += NcnnLinearOperator::current_thread_vulkan_dispatches() - initial_vulkan_dispatches;
    statistics.vulkan_attention_blocks += NcnnVulkanAttentionOperator::current_thread_blocks() - initial_vulkan_attention_blocks;
    const NcnnVulkanRuntimeCounters final_vulkan_runtime_counters = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
    statistics.vulkan_compute_submissions += final_vulkan_runtime_counters.compute_submissions - initial_vulkan_runtime_counters.compute_submissions;
    statistics.vulkan_async_submissions += final_vulkan_runtime_counters.asynchronous_submissions - initial_vulkan_runtime_counters.asynchronous_submissions;
    statistics.vulkan_batch_uploads += final_vulkan_runtime_counters.batch_uploads - initial_vulkan_runtime_counters.batch_uploads;
    statistics.vulkan_batch_downloads += final_vulkan_runtime_counters.batch_downloads - initial_vulkan_runtime_counters.batch_downloads;
    statistics.vulkan_auxiliary_uploads += final_vulkan_runtime_counters.auxiliary_uploads - initial_vulkan_runtime_counters.auxiliary_uploads;
    statistics.vulkan_auxiliary_upload_bytes += final_vulkan_runtime_counters.auxiliary_upload_bytes - initial_vulkan_runtime_counters.auxiliary_upload_bytes;
    statistics.vulkan_staging_slot_resizes += final_vulkan_runtime_counters.staging_slot_resizes - initial_vulkan_runtime_counters.staging_slot_resizes;
    statistics.vulkan_staging_slot_reuses += final_vulkan_runtime_counters.staging_slot_reuses - initial_vulkan_runtime_counters.staging_slot_reuses;
    statistics.vulkan_staging_slot_acquisitions += final_vulkan_runtime_counters.staging_slot_acquisitions - initial_vulkan_runtime_counters.staging_slot_acquisitions;
    statistics.vulkan_staging_slot_contentions += final_vulkan_runtime_counters.staging_slot_contentions - initial_vulkan_runtime_counters.staging_slot_contentions;
    statistics.vulkan_command_buffer_reuses += final_vulkan_runtime_counters.command_buffer_reuses - initial_vulkan_runtime_counters.command_buffer_reuses;
    statistics.vulkan_attention_qkv_rope_fusions += final_vulkan_runtime_counters.attention_qkv_rope_fusions - initial_vulkan_runtime_counters.attention_qkv_rope_fusions;
    statistics.vulkan_attention_qkv_ring_fusions += final_vulkan_runtime_counters.attention_qkv_ring_fusions - initial_vulkan_runtime_counters.attention_qkv_ring_fusions;
    statistics.vulkan_attention_decode_sdpa_fusions += final_vulkan_runtime_counters.attention_decode_sdpa_fusions - initial_vulkan_runtime_counters.attention_decode_sdpa_fusions;
    statistics.vulkan_kv_ring_appends += final_vulkan_runtime_counters.kv_ring_appends - initial_vulkan_runtime_counters.kv_ring_appends;
    statistics.vulkan_kv_ring_resizes += final_vulkan_runtime_counters.kv_ring_resizes - initial_vulkan_runtime_counters.kv_ring_resizes;
    statistics.vulkan_kv_ring_wrapped_views += final_vulkan_runtime_counters.kv_ring_wrapped_views - initial_vulkan_runtime_counters.kv_ring_wrapped_views;
    auto residency = state.memory_manager.record_execution(model.graph);
    if (!residency)
        return residency.error();
    return logits;
}

Result<std::vector<std::vector<float>>> CpuExecutor::execute_decode_batch(const CompiledModel& model, std::span<const CpuDecodeBatchEntry> entries, CpuDecodeBatchMetrics& metrics) const
{
    if (entries.empty())
        return Error{ErrorCode::InvalidArgument, "decode batch cannot be empty"};

    const size_t session_count = entries.size();
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
                CpuBatch& merged = scratch.staged_merged;
                if (!merge_rows_into(hidden, merged))
                {
                    return Error{ErrorCode::InternalError, "cannot merge staged hidden rows"};
                }
                rms_norm_batch_into(merged, model.weights.at(model.final_norm_weight), model.descriptor.norm_epsilon, scratch.staged_output);
                split_rows(scratch.staged_output, hidden);
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
                std::vector<int8_t>& prediction_states = entries.front().state->expert_scratch.prediction_states;
                prediction_states.assign(moe.experts.size(), int8_t{-1});
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    CpuSessionState& state = *entries[session_index].state;
                    if (model.expert_cache && layer.layer_id < state.layers.size())
                    {
                        for (uint32_t expert_id : state.layers[layer.layer_id].previous_expert_ids)
                        {
                            if (expert_id >= moe.experts.size())
                                continue;
                            const ExpertPlan& previous = moe.experts[expert_id];
                            if (previous.gate_up_weight == invalid_tensor_handle)
                                continue;
                            const TensorData& gate_up = model.weights.at(previous.gate_up_weight);
                            const TensorData& down = model.weights.at(previous.down_weight);
                            if (gate_up.mxfp4_file_storage || down.mxfp4_file_storage)
                            {
                                SessionStatistics& statistics = *entries[session_index].statistics;
                                ++statistics.expert_route_predictions;
                                int8_t& prediction_state = prediction_states[expert_id];
                                if (prediction_state < 0)
                                {
                                    const auto prediction = model.expert_cache->prefetch_pair(gate_up, down, layer.layer_id, previous.cache_key);
                                    prediction_state = prediction && prediction.value() ? int8_t{1} : int8_t{0};
                                }
                                const bool prediction_ready = prediction_state != 0;
                                if (prediction_ready)
                                {
                                    ++statistics.expert_route_prediction_cache_hits;
                                }
                                else
                                {
                                    ++statistics.expert_route_prediction_cache_misses;
                                }
                            }
                        }
                    }
                    const auto start = std::chrono::steady_clock::now();
                    const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution();
                    execute_attention_block_into(model.weights, layer.attention, model.descriptor.norm_epsilon, model.descriptor.kv_cache_dtype, entries[session_index].position_offset, state.layers[layer.layer_id], state.attention_scratch,
                                                 hidden[session_index], state.attention_scratch.output);
                    hidden[session_index].swap(state.attention_scratch.output);
                    entries[session_index].statistics->attention_time_microseconds += elapsed_microseconds(start);
                    record_vulkan_execution_delta(*entries[session_index].statistics, vulkan_before);
                }
                continue;
            }
            if (node->type == ExecutionNodeType::Router)
            {
                if (model.expert_cache)
                    model.expert_cache->cancel_prediction();
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                CpuBatch& merged_hidden = scratch.staged_merged;
                if (!merge_rows_into(hidden, merged_hidden))
                {
                    return Error{ErrorCode::InternalError, "cannot merge staged Router rows"};
                }
                CpuBatch& merged_normalized = scratch.staged_output;
                rms_norm_batch_into(merged_hidden, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, merged_normalized);
                CpuBatch& merged_logits = scratch.staged_router_logits;
                linear_batch_into(model.weights.at(moe.router_weight), merged_normalized, merged_logits);
                if (moe.router_bias != invalid_tensor_handle)
                {
                    add_bias_inplace(merged_logits, model.weights.at(moe.router_bias));
                }
                const auto router_start = std::chrono::steady_clock::now();
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    layer_state.normalized.reset(1, merged_normalized.columns(), false);
                    layer_state.router_logits.reset(1, merged_logits.columns(), false);
                    std::copy_n(merged_normalized.row(session_index), merged_normalized.columns(), layer_state.normalized.row(0));
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
                    options.normalization = moe.normalization;
                    options.flags = has_flag(moe.flags, MoeBlockNormalizeTopKWeights) ? static_cast<uint32_t>(ExpertDispatchNormalizeTopKWeights) : 0u;
                    auto dispatched = dispatcher.dispatch_into(layer_state.router_logits.values(), 1, options, layer_state.dispatch_plan);
                    if (!dispatched)
                        return dispatched.error();
                    const ExpertDispatchPlan& plan = layer_state.dispatch_plan;
                    statistics.expert_assignments += plan.assignment_count;
                    layer_state.active_experts.resize(plan.batches.size());
                    const std::vector<uint32_t>& previous_expert_ids = state.layers[layer.layer_id].previous_expert_ids;
                    for (const ExpertBatch& batch : plan.batches)
                    {
                        if (std::find(previous_expert_ids.begin(), previous_expert_ids.end(), batch.expert_id) != previous_expert_ids.end())
                        {
                            ++statistics.expert_route_prediction_matches;
                        }
                    }
                    state.layers[layer.layer_id].previous_expert_ids.clear();
                    state.layers[layer.layer_id].previous_expert_ids.reserve(plan.batches.size());
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
                        state.layers[layer.layer_id].previous_expert_ids.push_back(active.batch.expert_id);
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
                const auto engine_start = std::chrono::steady_clock::now();
                auto executed = run_moe(model, moe, combined, aggregate_statistics, entries.front().state->expert_scratch, layer.layer_id);
                const uint64_t engine_elapsed = elapsed_microseconds(engine_start);
                if (!executed)
                    return executed.error();

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

                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    SessionStatistics& statistics = *entries[session_index].statistics;
                    statistics.expert_engine_time_microseconds += engine_elapsed;
                    statistics.expert_compute_time_microseconds += aggregate_statistics.expert_compute_time_microseconds;
                    statistics.expert_cache_wait_time_microseconds += aggregate_statistics.expert_cache_wait_time_microseconds;
                    statistics.expert_cache_management_time_microseconds += aggregate_statistics.expert_cache_management_time_microseconds;
                    statistics.expert_regroup_time_microseconds += aggregate_statistics.expert_regroup_time_microseconds;
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
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    if (!layer_state.experts_executed)
                    {
                        return Error{ErrorCode::InternalError, "Combine executed before its Expert wave"};
                    }
                    const auto combine_start = std::chrono::steady_clock::now();
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
                    add_batch_inplace(hidden[session_index], moe_output);
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
