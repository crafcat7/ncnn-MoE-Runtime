#include "cpu_executor.h"

#include "kernels/cpu_fast_math.h"
#include "kernels/cpu_attention.h"
#include "kernels/cpu_batch.h"
#include "kernels/cpu_bfloat16.h"
#include "kernels/cpu_gated_delta_net.h"
#include "kernels/cpu_hyper_connection.h"
#include "kernels/cpu_latent_attention.h"
#include "kernels/cpu_ops.h"
#include "cpu_session_state.h"
#include "cpu_thread_budget.h"
#include "cpu_topology.h"
#include "execution_event_runtime.h"
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

class ScopedExpertBackendForeground
{
public:
    explicit ScopedExpertBackendForeground(
        const std::shared_ptr<IExpertExecutionBackend>& backend) noexcept
        : backend_(backend)
    {
        if (backend_)
            backend_->set_foreground_active(true);
    }

    ~ScopedExpertBackendForeground()
    {
        if (backend_)
            backend_->set_foreground_active(false);
    }

    ScopedExpertBackendForeground(const ScopedExpertBackendForeground&) = delete;
    ScopedExpertBackendForeground& operator=(const ScopedExpertBackendForeground&) = delete;

private:
    std::shared_ptr<IExpertExecutionBackend> backend_;
};

static bool use_fused_float8_gate_up(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(
        optimization_flags,
        RuntimeOptimizationCpuFloat8FusedGateUp);
}

static constexpr size_t expert_prefetch_limit_bytes = 4 * 1024;
static constexpr size_t assumed_cache_line_bytes = 64;

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

static VulkanExecutionSnapshot capture_vulkan_execution(
    const NcnnVulkanContextInstancePtr& context_instance)
{
    VulkanExecutionSnapshot snapshot;
    const NcnnVulkanExecutionSnapshot captured = NcnnLinearOperator::vulkan_execution_snapshot(context_instance);
    snapshot.dispatches = captured.dispatches;
    snapshot.attention_blocks = captured.attention_blocks;
    snapshot.counters = captured.counters;
    return snapshot;
}

static VulkanExecutionSnapshot vulkan_execution_delta(
    const VulkanExecutionSnapshot& before,
    const NcnnVulkanContextInstancePtr& context_instance)
{
    VulkanExecutionSnapshot delta;
    const NcnnVulkanExecutionSnapshot after_snapshot = NcnnLinearOperator::vulkan_execution_snapshot(context_instance);
    delta.dispatches = after_snapshot.dispatches - before.dispatches;
    delta.attention_blocks = after_snapshot.attention_blocks - before.attention_blocks;
    const NcnnVulkanRuntimeCounters& after = after_snapshot.counters;
    delta.counters.compute_submissions = after.compute_submissions - before.counters.compute_submissions;
    delta.counters.submit_wait_time_microseconds = after.submit_wait_time_microseconds
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
    delta.counters.command_graph_submissions = after.command_graph_submissions
                                               - before.counters.command_graph_submissions;
    delta.counters.command_graph_operations = after.command_graph_operations
                                              - before.counters.command_graph_operations;
    delta.counters.direct_host_input_bindings = after.direct_host_input_bindings - before.counters.direct_host_input_bindings;
    delta.counters.direct_host_output_bindings = after.direct_host_output_bindings - before.counters.direct_host_output_bindings;
    delta.counters.attention_qkv_rope_fusions = after.attention_qkv_rope_fusions - before.counters.attention_qkv_rope_fusions;
    delta.counters.attention_device_rope_fusions = after.attention_device_rope_fusions - before.counters.attention_device_rope_fusions;
    delta.counters.attention_qkv_ring_fusions = after.attention_qkv_ring_fusions - before.counters.attention_qkv_ring_fusions;
    delta.counters.attention_qkv_rope_pipeline_failures = after.attention_qkv_rope_pipeline_failures - before.counters.attention_qkv_rope_pipeline_failures;
    delta.counters.attention_qkv_rope_shape_failures = after.attention_qkv_rope_shape_failures - before.counters.attention_qkv_rope_shape_failures;
    delta.counters.attention_qkv_rope_source_failures = after.attention_qkv_rope_source_failures - before.counters.attention_qkv_rope_source_failures;
    delta.counters.attention_qkv_rope_norm_failures = after.attention_qkv_rope_norm_failures - before.counters.attention_qkv_rope_norm_failures;
    delta.counters.attention_qkv_rope_ring_failures = after.attention_qkv_rope_ring_failures - before.counters.attention_qkv_rope_ring_failures;
    delta.counters.attention_qkv_rope_allocation_failures = after.attention_qkv_rope_allocation_failures - before.counters.attention_qkv_rope_allocation_failures;
    delta.counters.attention_precondition_failures = after.attention_precondition_failures - before.counters.attention_precondition_failures;
    delta.counters.attention_staging_failures = after.attention_staging_failures - before.counters.attention_staging_failures;
    delta.counters.attention_norm_failures = after.attention_norm_failures - before.counters.attention_norm_failures;
    delta.counters.attention_qkv_failures = after.attention_qkv_failures - before.counters.attention_qkv_failures;
    delta.counters.attention_cache_failures = after.attention_cache_failures - before.counters.attention_cache_failures;
    delta.counters.attention_sdpa_failures = after.attention_sdpa_failures - before.counters.attention_sdpa_failures;
    delta.counters.attention_projection_failures = after.attention_projection_failures - before.counters.attention_projection_failures;
    delta.counters.attention_output_failures = after.attention_output_failures - before.counters.attention_output_failures;
    delta.counters.attention_submit_failures = after.attention_submit_failures - before.counters.attention_submit_failures;
    delta.counters.attention_decode_sdpa_fusions = after.attention_decode_sdpa_fusions - before.counters.attention_decode_sdpa_fusions;
    delta.counters.attention_cache_materializations = after.attention_cache_materializations
                                                      - before.counters.attention_cache_materializations;
    delta.counters.attention_cpu_fallbacks = after.attention_cpu_fallbacks
                                             - before.counters.attention_cpu_fallbacks;
    delta.counters.shared_expert_swiglu_fusions = after.shared_expert_swiglu_fusions - before.counters.shared_expert_swiglu_fusions;
    delta.counters.gated_delta_fusions = after.gated_delta_fusions - before.counters.gated_delta_fusions;
    delta.counters.gated_delta_submissions = after.gated_delta_submissions - before.counters.gated_delta_submissions;
    delta.counters.rms_norm_linear_fusions = after.rms_norm_linear_fusions - before.counters.rms_norm_linear_fusions;
    delta.counters.kv_ring_appends = after.kv_ring_appends - before.counters.kv_ring_appends;
    delta.counters.kv_ring_resizes = after.kv_ring_resizes - before.counters.kv_ring_resizes;
    delta.counters.kv_ring_wrapped_views = after.kv_ring_wrapped_views - before.counters.kv_ring_wrapped_views;
    delta.counters.kv_cache_promotions = after.kv_cache_promotions - before.counters.kv_cache_promotions;
    delta.counters.kv_cache_promotion_bytes = after.kv_cache_promotion_bytes - before.counters.kv_cache_promotion_bytes;
    delta.counters.bfloat16_cooperative_matrix_dispatches = after.bfloat16_cooperative_matrix_dispatches
                                                            - before.counters.bfloat16_cooperative_matrix_dispatches;
    delta.counters.command_dispatches = after.command_dispatches - before.counters.command_dispatches;
    delta.counters.command_pipeline_binds = after.command_pipeline_binds
                                            - before.counters.command_pipeline_binds;
    delta.counters.command_redundant_pipeline_binds = after.command_redundant_pipeline_binds
                                                      - before.counters.command_redundant_pipeline_binds;
    delta.counters.command_descriptor_bindings = after.command_descriptor_bindings
                                                 - before.counters.command_descriptor_bindings;
    delta.counters.command_push_constant_updates = after.command_push_constant_updates
                                                   - before.counters.command_push_constant_updates;
    delta.counters.command_resource_barrier_calls = after.command_resource_barrier_calls
                                                    - before.counters.command_resource_barrier_calls;
    delta.counters.command_buffer_resource_barriers = after.command_buffer_resource_barriers
                                                      - before.counters.command_buffer_resource_barriers;
    delta.counters.command_image_resource_barriers = after.command_image_resource_barriers
                                                     - before.counters.command_image_resource_barriers;
    return delta;
}

static void record_captured_vulkan_execution_delta(SessionStatistics& statistics, const VulkanExecutionSnapshot& delta)
{
    statistics.vulkan_linear_dispatches += delta.dispatches;
    statistics.vulkan_attention_blocks += delta.attention_blocks;
    statistics.vulkan_compute_submissions += delta.counters.compute_submissions;
    statistics.vulkan_submit_wait_time_microseconds += delta.counters.submit_wait_time_microseconds;
    statistics.vulkan_batch_uploads += delta.counters.batch_uploads;
    statistics.vulkan_batch_downloads += delta.counters.batch_downloads;
    statistics.vulkan_auxiliary_uploads += delta.counters.auxiliary_uploads;
    statistics.vulkan_auxiliary_upload_bytes += delta.counters.auxiliary_upload_bytes;
    statistics.vulkan_staging_slot_resizes += delta.counters.staging_slot_resizes;
    statistics.vulkan_staging_slot_reuses += delta.counters.staging_slot_reuses;
    statistics.vulkan_staging_slot_acquisitions += delta.counters.staging_slot_acquisitions;
    statistics.vulkan_staging_slot_contentions += delta.counters.staging_slot_contentions;
    statistics.vulkan_command_buffer_reuses += delta.counters.command_buffer_reuses;
    statistics.vulkan_command_graph_submissions += delta.counters.command_graph_submissions;
    statistics.vulkan_command_graph_operations += delta.counters.command_graph_operations;
    statistics.vulkan_direct_host_input_bindings += delta.counters.direct_host_input_bindings;
    statistics.vulkan_direct_host_output_bindings += delta.counters.direct_host_output_bindings;
    statistics.vulkan_attention_qkv_rope_fusions += delta.counters.attention_qkv_rope_fusions;
    statistics.vulkan_attention_device_rope_fusions += delta.counters.attention_device_rope_fusions;
    statistics.vulkan_attention_qkv_ring_fusions += delta.counters.attention_qkv_ring_fusions;
    statistics.vulkan_attention_qkv_rope_pipeline_failures += delta.counters.attention_qkv_rope_pipeline_failures;
    statistics.vulkan_attention_qkv_rope_shape_failures += delta.counters.attention_qkv_rope_shape_failures;
    statistics.vulkan_attention_qkv_rope_source_failures += delta.counters.attention_qkv_rope_source_failures;
    statistics.vulkan_attention_qkv_rope_norm_failures += delta.counters.attention_qkv_rope_norm_failures;
    statistics.vulkan_attention_qkv_rope_ring_failures += delta.counters.attention_qkv_rope_ring_failures;
    statistics.vulkan_attention_qkv_rope_allocation_failures += delta.counters.attention_qkv_rope_allocation_failures;
    statistics.vulkan_attention_precondition_failures += delta.counters.attention_precondition_failures;
    statistics.vulkan_attention_staging_failures += delta.counters.attention_staging_failures;
    statistics.vulkan_attention_norm_failures += delta.counters.attention_norm_failures;
    statistics.vulkan_attention_qkv_failures += delta.counters.attention_qkv_failures;
    statistics.vulkan_attention_cache_failures += delta.counters.attention_cache_failures;
    statistics.vulkan_attention_sdpa_failures += delta.counters.attention_sdpa_failures;
    statistics.vulkan_attention_projection_failures += delta.counters.attention_projection_failures;
    statistics.vulkan_attention_output_failures += delta.counters.attention_output_failures;
    statistics.vulkan_attention_submit_failures += delta.counters.attention_submit_failures;
    statistics.vulkan_attention_decode_sdpa_fusions += delta.counters.attention_decode_sdpa_fusions;
    statistics.vulkan_attention_cache_materializations += delta.counters.attention_cache_materializations;
    statistics.vulkan_attention_cpu_fallbacks += delta.counters.attention_cpu_fallbacks;
    statistics.vulkan_shared_expert_swiglu_fusions += delta.counters.shared_expert_swiglu_fusions;
    statistics.vulkan_gated_delta_fusions += delta.counters.gated_delta_fusions;
    statistics.vulkan_gated_delta_submissions += delta.counters.gated_delta_submissions;
    statistics.vulkan_rms_norm_linear_fusions += delta.counters.rms_norm_linear_fusions;
    statistics.vulkan_kv_ring_appends += delta.counters.kv_ring_appends;
    statistics.vulkan_kv_ring_resizes += delta.counters.kv_ring_resizes;
    statistics.vulkan_kv_ring_wrapped_views += delta.counters.kv_ring_wrapped_views;
    statistics.vulkan_kv_cache_promotions += delta.counters.kv_cache_promotions;
    statistics.vulkan_kv_cache_promotion_bytes += delta.counters.kv_cache_promotion_bytes;
    statistics.vulkan_bfloat16_cooperative_matrix_dispatches += delta.counters.bfloat16_cooperative_matrix_dispatches;
    statistics.vulkan_command_dispatches += delta.counters.command_dispatches;
    statistics.vulkan_command_pipeline_binds += delta.counters.command_pipeline_binds;
    statistics.vulkan_command_redundant_pipeline_binds += delta.counters.command_redundant_pipeline_binds;
    statistics.vulkan_command_descriptor_bindings += delta.counters.command_descriptor_bindings;
    statistics.vulkan_command_push_constant_updates += delta.counters.command_push_constant_updates;
    statistics.vulkan_command_resource_barrier_calls += delta.counters.command_resource_barrier_calls;
    statistics.vulkan_command_buffer_resource_barriers += delta.counters.command_buffer_resource_barriers;
    statistics.vulkan_command_image_resource_barriers += delta.counters.command_image_resource_barriers;
}

static void record_vulkan_execution_delta(
    SessionStatistics& statistics,
    const VulkanExecutionSnapshot& before,
    const NcnnVulkanContextInstancePtr& context_instance)
{
    record_captured_vulkan_execution_delta(
        statistics,
        vulkan_execution_delta(before, context_instance));
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
    statistics.expert_gpu_route_aggregation_batches += after.route_aggregation_batches - before.route_aggregation_batches;
    statistics.expert_gpu_route_aggregation_routes += after.route_aggregation_routes - before.route_aggregation_routes;
    statistics.expert_gpu_route_aggregation_bytes_saved += after.route_aggregation_bytes_saved - before.route_aggregation_bytes_saved;
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
        else if (has_flag(expert.flags, ExpertPlanPackedGateUp))
        {
            CpuBatch gate_up = expert_linear(gate_up_weight, gate_up_bias, cached_weights ? nullptr : &operators.at_weight(expert.gate_up_weight), input, metrics, optimization_flags);
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
            CpuBatch gate_up = expert_linear(gate_up_weight, gate_up_bias, cached_weights ? nullptr : &operators.at_weight(expert.gate_up_weight), input, metrics, optimization_flags);
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
        if (prefetch)
            metrics.hinted_bytes += prefetch_tensor(down_weight);
        return expert_linear(down_weight, expert.down_bias == invalid_tensor_handle ? nullptr : &weights.at(expert.down_bias), cached_weights ? nullptr : &operators.at_weight(expert.down_weight), activated, metrics, optimization_flags);
    }

    if (prefetch)
        metrics.hinted_bytes += prefetch_weight(weights, expert.up_weight);
    const TensorData& up_weight = weights.at(expert.up_weight);
    bool gate_prefetched = false;
    if (has_flag(expert.flags, ExpertPlanGated) && expert.gate_weight != invalid_tensor_handle)
    {
        const TensorData& gate_weight = weights.at(expert.gate_weight);
        const TensorData& down_weight = weights.at(expert.down_weight);
        const CompiledOperator& gate_operator = operators.at_weight(expert.gate_weight);
        const CompiledOperator& up_operator = operators.at_weight(expert.up_weight);
        const CompiledOperator& down_operator = operators.at_weight(expert.down_weight);
        if (use_fused_float8_gate_up(optimization_flags)
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
                    &gate_operator,
                    &up_operator))
            {
                if (prefetch)
                {
                    metrics.hinted_bytes += prefetch_weight(weights, expert.down_weight);
                }
                return expert_linear(down_weight, nullptr, &down_operator, activated, metrics, optimization_flags);
            }
        }
    }
    CpuBatch up = expert_linear(up_weight, nullptr, &operators.at_weight(expert.up_weight), input, metrics, optimization_flags);
    if (has_flag(expert.flags, ExpertPlanGated))
    {
        if (prefetch && !gate_prefetched)
            metrics.hinted_bytes += prefetch_weight(weights, expert.gate_weight);
        const CpuBatch gate = expert_linear(weights.at(expert.gate_weight), nullptr, &operators.at_weight(expert.gate_weight), input, metrics, optimization_flags);
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
    return expert_linear(weights.at(expert.down_weight), nullptr, &operators.at_weight(expert.down_weight), up, metrics, optimization_flags);
}

static CpuBatch run_shared_expert(
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
                    &model.operators.at_weight(expert.down_weight),
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
        &model.operators.at_weight(moe.shared_expert_gate_weight));
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

static void gather_tokens(const CpuBatch& source, const std::vector<ExpertRoute>& routes, CpuBatch& gathered)
{
    gathered.reset(routes.size(), source.columns(), false);
    for (size_t route_index = 0; route_index < routes.size(); ++route_index)
    {
        std::copy_n(source.row(routes[route_index].token_index), source.columns(), gathered.row(route_index));
    }
}

static CpuBatch run_expert(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const ExpertPlan& expert,
    const ExpertCacheLease* cached_weights,
    const CpuBatch& input,
    bool prefetch,
    ExpertExecutionMetrics& metrics,
    uint64_t optimization_flags);

struct HybridBlockState
{
    size_t active_index = 0;
    size_t input_begin = 0;
    CpuBatch input;
    CpuBatch output;
    bool gpu_planned = false;
    bool gpu_executed = false;
    bool cpu_needed = false;
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
            && active_ready[block.active_index] != 0
            && block.cpu_needed
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
    team_size = std::min(static_cast<int>(groups.size()), static_cast<int>(cpu_linear_thread_limit()));
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
                model.optimization_flags);
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
            model.optimization_flags);
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
                moe.experts[active.batch.expert_id].weight_bytes,
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
            || backend_aggregated[block.active_index] != 0
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

    const bool grouped_decode = mxfp4_expert_batch(decode_tasks, &scratch.kernels, model.optimization_flags);
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
    expert_team_size = std::min(static_cast<int>(active_indices.size()), static_cast<int>(cpu_linear_thread_limit()));
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
        active.output = run_expert(model.weights, model.operators, moe.experts[expert_id], active.lease.gate_up ? &active.lease : nullptr, active.input, prefetch, active.metrics, model.optimization_flags);
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
    return (expert.activation == ExpertActivation::Silu
            || expert.activation == ExpertActivation::GptOssSwiGlu
            || expert.activation == ExpertActivation::DeepSeekSwiGlu)
           && gate_up.dtype == DType::MxFp4
           && down.dtype == DType::MxFp4
           && gate_up.shape.size() == 2
           && down.shape.size() == 2
           && gate_up.shape[0] % 2 == 0
           && down.shape[1] == gate_up.shape[0] / 2;
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
        || !can_run_vulkan_expert(expert, *lease.gate_up, *lease.down))
    {
        return;
    }
    if (token_count < vulkan_expert_gpu_admission_min_rows)
        return;
    model.expert_backend->admit(expert.cache_key, lease.gate_up, expert.gate_up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.gate_up_bias), lease.down,
                                expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias), residency_group, token_count,
                                expert.activation_limit, expert.activation);
}

struct SpeculativeLayerExecutionNodes
{
    uint32_t layer_plan_index = invalid_execution_layer_id;
    const ExecutionNode* attention = nullptr;
    const ExecutionNode* router = nullptr;
    const ExecutionNode* expert_dispatch = nullptr;
    const ExecutionNode* expert_group = nullptr;
    const ExecutionNode* shared_expert_group = nullptr;
    const ExecutionNode* combine = nullptr;
};

static Result<std::vector<SpeculativeLayerExecutionNodes>>
collect_speculative_layer_execution_nodes(const CompiledModel& model)
{
    const ExecutionGraph& graph = model.speculative.graph;
    std::vector<SpeculativeLayerExecutionNodes> layers(
        graph.layer_plans.size());
    for (ExecutionNodeId node_id : model.speculative.schedule.node_order)
    {
        if (node_id >= graph.nodes.size())
        {
            return Error{
                ErrorCode::InternalError,
                "speculative schedule references an invalid node"};
        }
        const ExecutionNode* node = &graph.nodes[node_id];
        if (node->layer_plan_index == invalid_execution_layer_id)
            continue;
        if (node->layer_plan_index >= layers.size())
        {
            return Error{
                ErrorCode::InternalError,
                "speculative node layer binding is out of range"};
        }
        SpeculativeLayerExecutionNodes& layer = layers[node->layer_plan_index];
        if (layer.layer_plan_index == invalid_execution_layer_id)
            layer.layer_plan_index = node->layer_plan_index;
        const ExecutionNode** target = nullptr;
        switch (node->type)
        {
        case ExecutionNodeType::Attention:
            target = &layer.attention;
            break;
        case ExecutionNodeType::Router:
            target = &layer.router;
            break;
        case ExecutionNodeType::ExpertDispatch:
            target = &layer.expert_dispatch;
            break;
        case ExecutionNodeType::ExpertGroup:
            target = &layer.expert_group;
            break;
        case ExecutionNodeType::SharedExpertGroup:
            target = &layer.shared_expert_group;
            break;
        case ExecutionNodeType::Combine:
            target = &layer.combine;
            break;
        default:
            break;
        }
        if (target && *target != nullptr)
        {
            return Error{
                ErrorCode::InternalError,
                "speculative graph contains duplicate layer nodes"};
        }
        if (target)
            *target = node;
    }

    for (const SpeculativeLayerExecutionNodes& layer : layers)
    {
        if (!layer.attention || !layer.router || !layer.expert_dispatch
            || !layer.expert_group || !layer.combine)
        {
            return Error{
                ErrorCode::InternalError,
                "speculative graph is missing a required layer node"};
        }
    }
    return layers;
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
                model.weights.at(expert.down_weight)))
        {
            return false;
        }
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

    // Establish the same gathered-input invariant as the legacy path.
    for (ActiveExpertExecution& active : layer_state.active_experts)
    {
        gather_tokens(layer_state.normalized, active.batch.routes, active.input);
        if (active.failed)
            return active.error;
    }

    size_t total_routes = 0;
    for (const ActiveExpertExecution& active : layer_state.active_experts)
        total_routes += active.batch.routes.size();

    std::vector<HybridBlockState> blocks;
    blocks.reserve(total_routes);
    scratch.backend_aggregated.assign(active_expert_count, 0);
    scratch.backend_aggregated_output.reset(
        layer_state.normalized.rows(),
        model.descriptor.hidden_size,
        true);
    std::vector<ExpertBackendRequest> requests;
    requests.reserve(total_routes);
    uint8_t route_aggregation_completed = 0;
    std::vector<uint8_t> cpu_ready(active_expert_count, 0);
    std::vector<size_t> cpu_active_indices;
    cpu_active_indices.reserve(active_expert_count);

    // Build contiguous route blocks from the dispatcher order.
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

    const auto add_cpu_work = [&](size_t active_index) {
        if (cpu_ready[active_index] == 0)
        {
            cpu_ready[active_index] = 2;
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
                                          victim_metadata(model, expert)});
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
                const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                if (expert.runtime)
                {
                    if (active.lease.cache_hit)
                        expert.runtime->record_cache_hit();
                    else
                        expert.runtime->record_cache_miss();
                    expert.runtime->set_residency(ExpertCacheState::Resident, TensorLocation::Cpu);
                }
            }
            active_indices.swap(remaining);
        }
        return {};
    };

    for (size_t block_index = 0; block_index < blocks.size(); ++block_index)
    {
        HybridBlockState& block = blocks[block_index];
        const ActiveExpertExecution& active = layer_state.active_experts[block.active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        ExpertBackendRequest request{
            expert.cache_key,
            &block.input,
            &block.output,
            expert.weight_bytes};
        request.route_aggregation.output = &scratch.backend_aggregated_output;
        request.route_aggregation.routes = std::span<const ExpertRoute>(active.batch.routes).subspan(block.input_begin, block.input.rows());
        request.route_aggregation.token_count = static_cast<uint32_t>(layer_state.normalized.rows());
        request.route_aggregation.completed = &route_aggregation_completed;
        request.route_aggregation.require_all_requests = true;
        requests.push_back(request);
    }

    const auto backend_execution_start = std::chrono::steady_clock::now();
    std::unique_ptr<IExpertBackendBatchSubmission> submission = model.expert_backend->submit_batch(requests);
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
            backend_total_weight_bytes += expert.weight_bytes;
            active_accelerated[block.active_index] = 2;
        }
        if (planned[request_index] == ExpertBackendExecutionResult::Executed)
        {
            block.gpu_planned = true;
        }
        else
        {
            block.cpu_needed = true;
            add_cpu_work(block.active_index);
        }
    }

    // Acquire host weights only for CPU blocks.
    std::vector<size_t> pending_cpu;
    pending_cpu.reserve(cpu_active_indices.size());
    for (size_t active_index : cpu_active_indices)
    {
        ActiveExpertExecution& active = layer_state.active_experts[active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        const TensorData* gate_up = expert.gate_up_weight == invalid_tensor_handle
                                        ? nullptr
                                        : &model.weights.at(expert.gate_up_weight);
        const TensorData& down = model.weights.at(expert.down_weight);
        if (!model.expert_cache || !gate_up
            || (!gate_up->mxfp4_file_storage && !down.mxfp4_file_storage))
        {
            cpu_ready[active_index] = 1;
            continue;
        }
        pending_cpu.push_back(active_index);
    }
    if (!pending_cpu.empty())
    {
        auto acquired = acquire_cpu_weights(pending_cpu);
        if (!acquired)
        {
            submission->abort();
            return acquired.error();
        }
    }
    statistics.expert_cache_management_time_microseconds += elapsed_microseconds(cache_management_start);

    const uint64_t initial_cpu_time = run_hybrid_cpu_blocks(
        model,
        moe,
        layer_state,
        blocks,
        cpu_ready,
        prefetch);
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
            if (active_accelerated[block.active_index] == 2)
            {
                active_accelerated[block.active_index] = 1;
                backend_accelerated_weight_bytes += moe.experts[layer_state.active_experts[block.active_index].batch.expert_id].weight_bytes;
            }
        }
        else
        {
            block.gpu_planned = false;
            block.cpu_needed = true;
            add_cpu_work(block.active_index);
        }
    }

    // Aggregation is published only after every block completes.
    if (committed && route_aggregation_completed != 0)
        std::fill(
            scratch.backend_aggregated.begin(),
            scratch.backend_aggregated.end(),
            uint8_t{1});

    // Retry failed blocks on CPU without recomputing completed blocks.
    pending_cpu.clear();
    for (size_t active_index : cpu_active_indices)
    {
        if (cpu_ready[active_index] == 1)
            continue;
        const ActiveExpertExecution& active = layer_state.active_experts[active_index];
        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
        const TensorData* gate_up = expert.gate_up_weight == invalid_tensor_handle
                                        ? nullptr
                                        : &model.weights.at(expert.gate_up_weight);
        const TensorData& down = model.weights.at(expert.down_weight);
        if (!model.expert_cache || !gate_up
            || (!gate_up->mxfp4_file_storage && !down.mxfp4_file_storage))
        {
            cpu_ready[active_index] = 1;
            continue;
        }
        pending_cpu.push_back(active_index);
    }
    if (!pending_cpu.empty())
    {
        auto acquired = acquire_cpu_weights(pending_cpu);
        if (!acquired)
            return acquired.error();
    }
    const uint64_t fallback_cpu_time = run_hybrid_cpu_blocks(
        model,
        moe,
        layer_state,
        blocks,
        cpu_ready,
        prefetch);
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
        const std::span<uint32_t> ranked = std::span<uint32_t>(actual_storage).first(layer.moe.top_k);
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
         layer_index < model.graph.layer_plans.size();
         ++layer_index)
    {
        const CompiledLayerPlan& candidate = model.graph.layer_plans[layer_index];
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
        predicted_logits,
        model.optimization_flags,
        &model.operators.at_weight(next_layer.moe.router_weight));
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
        options.selection_bias = model.weights.at(next_layer.moe.router_selection_bias).float32_values();
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
        const std::span<uint32_t> ranked = std::span<uint32_t>(predicted_storage).first(next_layer.moe.top_k);
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
    state.layers[outcome.target_layer_id].predicted_expert_ids = std::move(outcome.predicted_expert_ids);
    statistics.expert_route_prediction_cache_hits += outcome.cache_hits;
    statistics.expert_route_prediction_cache_misses += outcome.cache_misses;
    statistics.expert_route_prediction_time_microseconds += outcome.predictor_time_microseconds;
    return {};
}

static void admit_ready_router_prediction(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    CpuSessionState& state)
{
    if (!model.expert_cache
        || !model.expert_backend
        || layer.layer_id >= state.layers.size())
    {
        return;
    }

    CpuLayerCache& cache = state.layers[layer.layer_id];
    if (cache.predicted_expert_ids.empty())
        return;

    std::array<uint32_t, maximum_expert_route_ranks> admitted_ids{};
    size_t admitted_count = 0;
    for (const uint32_t expert_id : cache.predicted_expert_ids)
    {
        if (expert_id == std::numeric_limits<uint32_t>::max()
            || expert_id >= layer.moe.experts.size()
            || admitted_count == admitted_ids.size()
            || std::find(
                   admitted_ids.begin(),
                   admitted_ids.begin() + admitted_count,
                   expert_id)
                   != admitted_ids.begin() + admitted_count)
        {
            continue;
        }

        const ExpertPlan& expert = layer.moe.experts[expert_id];
        if (expert.gate_up_weight == invalid_tensor_handle
            || expert.down_weight == invalid_tensor_handle)
        {
            continue;
        }
        const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
        const TensorData& down = model.weights.at(expert.down_weight);
        if (!gate_up.mxfp4_file_storage
            || !down.mxfp4_file_storage
            || !can_run_vulkan_expert(expert, gate_up, down))
        {
            continue;
        }

        const ExpertCachePairRequest request{
            &gate_up,
            &down,
            layer.layer_id,
            expert.cache_key,
            victim_metadata(model, expert)};
        ExpertCacheLease lease;
        auto ready = model.expert_cache->try_acquire_ready_pairs(
            std::span<const ExpertCachePairRequest>(&request, 1),
            std::span<ExpertCacheLease>(&lease, 1));
        if (!ready || !ready.value())
            continue;

        model.expert_backend->admit(
            expert.cache_key,
            std::move(lease.gate_up),
            expert.gate_up_bias == invalid_tensor_handle
                ? nullptr
                : &model.weights.at(expert.gate_up_bias),
            std::move(lease.down),
            expert.down_bias == invalid_tensor_handle
                ? nullptr
                : &model.weights.at(expert.down_bias),
            layer.layer_id,
            1,
            expert.activation_limit,
            expert.activation);
        admitted_ids[admitted_count++] = expert_id;
    }
}

static Result<void> complete_router_prediction(
    const CompiledModel& model,
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
    statistics.expert_route_prediction_wait_time_microseconds += elapsed_microseconds(wait_started);
    pending.target_layer_id = std::numeric_limits<uint32_t>::max();
    if (!completed)
        return completed.error();
    ++statistics.expert_route_prediction_async_completions;
    auto applied = apply_router_prediction(
        std::move(completed).value(),
        state,
        statistics);
    if (!applied)
        return applied.error();

    if (layer_id < model.graph.layer_plans.size())
    {
        // Admit predicted weights to the GPU queue after the host pair is ready.
        admit_ready_router_prediction(
            model,
            model.graph.layer_plans[layer_id],
            state);
    }
    return {};
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
                state.router_prediction_worker = std::make_unique<CpuTaskWorker>(2);
            }
            CpuBatch copied_input = router_input;
            auto promise = std::make_shared<
                std::promise<Result<RouterPredictionOutcome>>>();
            std::future<Result<RouterPredictionOutcome>> future = promise->get_future();
            const CompiledLayerPlan* next_layer = target.layer;
            const uint32_t prefetch_width = target.prefetch_width;
            Bfloat16BatchedLinearExecutionCounter* const bfloat16_counter = current_bfloat16_batched_linear_execution_counter();
            submitted = state.router_prediction_worker->try_submit(
                [&model,
                 next_layer,
                 copied_input = std::move(copied_input),
                 prefetch_width,
                 bfloat16_counter,
                 promise]() mutable {
                    const ScopedBfloat16BatchedLinearExecutionCounter
                        bfloat16_scope(bfloat16_counter);
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
    auto applied = apply_router_prediction(
        std::move(outcome).value(),
        state,
        statistics);
    if (!applied)
        return applied.error();
    admit_ready_router_prediction(model, *target.layer, state);
    return {};
}

static Result<void> run_moe(
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
    expert_team_size = std::min(static_cast<int>(active_expert_count), static_cast<int>(cpu_linear_thread_limit()));
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
    std::vector<uint8_t>& backend_aggregated = scratch.backend_aggregated;
    std::vector<size_t>& backend_indices = scratch.backend_indices;
    std::vector<ExpertBackendRequest>& backend_requests = scratch.backend_requests;
    backend_executed.assign(active_expert_count, 0);
    backend_aggregated.assign(active_expert_count, 0);
    scratch.backend_aggregated_output.reset(layer_state.normalized.rows(), model.descriptor.hidden_size, true);
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
    bool backend_reservation_shape_valid = true;
    if (backend == ExecutionBackend::Vulkan && model.expert_backend)
    {
        for (size_t active_index = 0; active_index < active_expert_count; ++active_index)
        {
            ActiveExpertExecution& active = layer_state.active_experts[active_index];
            const ExpertPlan& expert = moe.experts[active.batch.expert_id];
            if (expert.gate_up_weight == invalid_tensor_handle
                || !can_run_vulkan_expert(
                    expert,
                    model.weights.at(expert.gate_up_weight),
                    model.weights.at(expert.down_weight)))
            {
                continue;
            }
            backend_indices.push_back(active_index);
            backend_max_token_count = std::max<uint32_t>(backend_max_token_count, static_cast<uint32_t>(active.input.rows()));
            backend_total_weight_bytes += expert.weight_bytes;
            ExpertBackendRequest request{expert.cache_key, &active.input, &active.output, expert.weight_bytes};
            request.route_aggregation.output = &scratch.backend_aggregated_output;
            request.route_aggregation.routes = active.batch.routes;
            request.route_aggregation.token_count = static_cast<uint32_t>(layer_state.normalized.rows());
            request.route_aggregation.completed = &backend_aggregated[active_index];
            backend_requests.push_back(request);
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
                    backend_accelerated_weight_bytes += moe.experts[active.batch.expert_id].weight_bytes;
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
                admit_vulkan_expert(
                    model,
                    expert,
                    active.lease,
                    residency_group,
                    static_cast<uint32_t>(active.input.rows()),
                    backend);
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
            admit_vulkan_expert(
                model,
                expert,
                active.lease,
                residency_group,
                static_cast<uint32_t>(active.input.rows()),
                backend);
            if (expert.runtime)
            {
                expert.runtime->set_residency(ExpertCacheState::Resident, TensorLocation::Cpu);
            }
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
                    admit_vulkan_expert(
                        model,
                        expert,
                        active.lease,
                        residency_group,
                        static_cast<uint32_t>(active.input.rows()),
                        backend);
                    if (expert.runtime)
                    {
                        expert.runtime->set_residency(ExpertCacheState::Resident, TensorLocation::Cpu);
                    }
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

static bool initialize_backend_aggregated_output(
    const CpuExpertExecutionScratch& scratch,
    std::span<const ActiveExpertExecution> active_experts,
    size_t rows,
    uint32_t columns,
    CpuBatch& output)
{
    bool aggregated = false;
    const size_t count = std::min(active_experts.size(), scratch.backend_aggregated.size());
    for (size_t index = 0; index < count; ++index)
    {
        if (scratch.backend_aggregated[index] != 0)
        {
            aggregated = true;
            break;
        }
    }
    if (!aggregated
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

static Result<void> execute_speculative_layer(
    const CompiledModel& model,
    const SpeculativeLayerExecutionNodes& execution,
    uint64_t position_offset,
    CpuLayerCache& cache,
    CpuBatch& hidden,
    SessionStatistics& statistics,
    CpuExpertExecutionScratch& scratch,
    CpuAttentionExecutionScratch& attention_scratch)
{
    if (execution.layer_plan_index >= model.speculative.graph.layer_plans.size()
        || !execution.attention || !execution.router
        || !execution.expert_dispatch || !execution.expert_group
        || !execution.combine)
    {
        return Error{
            ErrorCode::InternalError,
            "speculative execution graph has an incomplete layer binding"};
    }
    const CompiledLayerPlan& layer = model.speculative.graph.layer_plans[execution.layer_plan_index];
    const ExecutionBackend attention_backend = execution.attention->backend;
    const ExecutionBackend expert_backend = execution.expert_group->backend;
    const bool cpu_prefetch = has_flag(execution.expert_group->flags, ExecutionNodeCpuPrefetch);

    LayerGraphState layer_state;
    const uint32_t multiplier = model.descriptor.hyper_connection_multiplier;
    const auto attention_start = std::chrono::steady_clock::now();
    CpuHyperConnectionMix attention_mix;
    const CpuBatch* attention_input = &hidden;
    if (multiplier > 1)
    {
        auto mixed = hyper_connection_pre(hidden, model.weights.at(layer.hyper_connection.attention_function), model.weights.at(layer.hyper_connection.attention_scale),
                                          model.weights.at(layer.hyper_connection.attention_base), multiplier, model.descriptor.hyper_connection_iterations,
                                          model.descriptor.norm_epsilon, model.descriptor.hyper_connection_epsilon,
                                          model.optimization_flags);
        if (!mixed)
            return mixed.error();
        attention_mix = std::move(mixed).value();
        attention_input = &attention_mix.reduced;
    }
    if (model.speculative.kind == SpeculativeModelKind::Mtp)
    {
        auto attention = execute_attention_block_into(
            model.weights,
            model.operators,
            layer.attention,
            attention_backend,
            model.descriptor.norm_epsilon,
            model.descriptor.kv_cache_dtype,
            position_offset,
            cache,
            attention_scratch,
            *attention_input,
            attention_scratch.output,
            model.optimization_flags);
        if (!attention)
            return attention.error();
        hidden.swap(attention_scratch.output);
    }
    else
    {
        auto attention = execute_dspark_attention(
            model.weights,
            model.operators,
            layer.attention,
            attention_backend,
            model.descriptor.norm_epsilon,
            position_offset,
            cache,
            *attention_input,
            model.optimization_flags);
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
                                          model.descriptor.norm_epsilon, model.descriptor.hyper_connection_epsilon,
                                          model.optimization_flags);
        if (!mixed)
            return mixed.error();
        layer_state.ffn_hyper_mix = std::move(mixed).value();
        rms_norm_batch_into(layer_state.ffn_hyper_mix.reduced, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset, model.optimization_flags);
    }
    else
    {
        rms_norm_batch_into(hidden, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset, model.optimization_flags);
    }
    linear_batch_into(
        model.weights.at(moe.router_weight),
        layer_state.normalized,
        layer_state.router_logits,
        model.optimization_flags,
        &model.operators.at_weight(moe.router_weight));
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

    const auto expert_engine_start = std::chrono::steady_clock::now();
    auto executed = run_moe(
        model,
        moe,
        layer_state,
        statistics,
        scratch,
        layer.layer_id,
        expert_backend,
        cpu_prefetch);
    statistics.expert_engine_time_microseconds += elapsed_microseconds(expert_engine_start);
    if (!executed)
        return executed.error();
    const auto combine_start = std::chrono::steady_clock::now();
    if (moe.has_shared_expert && layer_state.shared_expert_output.rows() == 0)
    {
        ExpertExecutionMetrics shared_metrics;
        layer_state.shared_expert_output = run_shared_expert(
            model,
            moe,
            layer_state.normalized,
            shared_metrics,
            model.optimization_flags);
    }
    CpuBatch& moe_output = layer_state.normalized;
    const bool has_backend_aggregation = initialize_backend_aggregated_output(
        scratch,
        layer_state.active_experts,
        hidden.rows(),
        model.descriptor.hidden_size,
        moe_output);
    for (size_t active_index = 0; active_index < layer_state.active_experts.size(); ++active_index)
    {
        const ActiveExpertExecution& active = layer_state.active_experts[active_index];
        if (has_backend_aggregation
            && active_index < scratch.backend_aggregated.size()
            && scratch.backend_aggregated[active_index] != 0)
        {
            continue;
        }
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
        model.descriptor.norm_weight_offset,
        model.optimization_flags);
    CpuBatch normalized_hidden = rms_norm_batch(
        target_hidden,
        model.weights.at(model.speculative.mtp_hidden_norm_weight),
        model.descriptor.norm_epsilon,
        model.descriptor.norm_weight_offset,
        model.optimization_flags);
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
        packed,
        model.optimization_flags,
        &model.operators.at_weight(model.speculative.mtp_input_projection_weight));
}

static Result<CpuBatch> execute_mtp_batch(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    const CpuBatch& target_hidden,
    uint64_t position_offset,
    SessionStatistics& statistics,
    CpuSessionState& state)
{
    auto layer_nodes = collect_speculative_layer_execution_nodes(model);
    if (!layer_nodes)
        return layer_nodes.error();
    if (layer_nodes.value().size() != 1
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
        layer_nodes.value().front(),
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
        model.descriptor.norm_weight_offset,
        model.optimization_flags);
}

static Result<void> append_mtp_context(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    const CpuBatch& target_hidden,
    uint64_t position_offset,
    CpuSessionState& state)
{
    auto layer_nodes = collect_speculative_layer_execution_nodes(model);
    if (!layer_nodes)
        return layer_nodes.error();
    if (layer_nodes.value().size() != 1
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
    const SpeculativeLayerExecutionNodes& execution = layer_nodes.value().front();
    const CompiledLayerPlan& layer = model.speculative.graph.layer_plans[execution.layer_plan_index];
    auto appended = append_attention_context_into(
        model.weights,
        model.operators,
        layer.attention,
        execution.attention->backend,
        model.descriptor.norm_epsilon,
        model.descriptor.kv_cache_dtype,
        position_offset,
        state.speculative_layers.front(),
        state.attention_scratch,
        prepared.value(),
        model.optimization_flags);
    return appended;
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

    const bool direct_alignment = !state.speculative_direct_alignment_ids.empty();
    const std::vector<int32_t>& input_ids = direct_alignment
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

    const bool has_pending = state.mtp_pending_target_hidden.rows() != 0;
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

    const size_t aligned_rows = direct_alignment
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
        uint64_t aligned_position = state.speculative_main_hidden_position;
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
    state.mtp_pending_target_position = state.speculative_main_hidden_position
                                        + target_hidden.rows() - 1;
    state.speculative_input_ids.clear();
    state.speculative_direct_alignment_ids.clear();
    return {};
}

Result<void> CpuExecutor::update_speculative_context(const CompiledModel& model, SessionStatistics& statistics, CpuSessionState& state) const
{
    if (!model.speculative.enabled()
        || !state.speculative_context_enabled)
        return {};
    Bfloat16BatchedLinearExecutionCounter cpu_bfloat16_execution;
    const ScopedBfloat16BatchedLinearExecutionCounter cpu_bfloat16_scope(
        &cpu_bfloat16_execution);
    const auto started = std::chrono::steady_clock::now();
    const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution(model.vulkan_context_instance);
    if (model.speculative.kind == SpeculativeModelKind::Mtp)
    {
        auto updated = update_mtp_context(
            model,
            state);
        if (!updated)
            return updated.error();
        record_vulkan_execution_delta(statistics, vulkan_before, model.vulkan_context_instance);
        statistics.cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_execution.dispatch_count();
        statistics.speculative_context_time_microseconds += elapsed_microseconds(started);
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
    CpuBatch projected = linear_batch(
        model.weights.at(model.speculative.main_projection_weight),
        state.speculative_main_hidden,
        model.optimization_flags,
        &model.operators.at_weight(model.speculative.main_projection_weight));
    projected = rms_norm_batch(projected, model.weights.at(model.speculative.main_norm_weight), model.descriptor.norm_epsilon, model.descriptor.norm_weight_offset, model.optimization_flags);
    auto layer_nodes = collect_speculative_layer_execution_nodes(model);
    if (!layer_nodes)
        return layer_nodes.error();
    if (state.speculative_layers.size() != layer_nodes.value().size())
        state.speculative_layers.resize(layer_nodes.value().size());
    for (size_t layer_index = 0; layer_index < layer_nodes.value().size(); ++layer_index)
    {
        const SpeculativeLayerExecutionNodes& execution = layer_nodes.value()[layer_index];
        const CompiledLayerPlan& layer = model.speculative.graph.layer_plans[execution.layer_plan_index];
        auto appended = append_dspark_attention_context(
            model.weights,
            model.operators,
            layer.attention,
            execution.attention->backend,
            model.descriptor.norm_epsilon,
            state.speculative_main_hidden_position, state.speculative_layers[layer_index], projected,
            model.optimization_flags);
        if (!appended)
            return appended.error();
    }
    record_vulkan_execution_delta(statistics, vulkan_before, model.vulkan_context_instance);
    statistics.cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_execution.dispatch_count();
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
    const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution(model.vulkan_context_instance);
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
    CpuBatch previous_hidden = state.mtp_pending_target_hidden;
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
            mtp_hidden.value(),
            model.optimization_flags,
            &model.operators.at_weight(model.lm_head_weight));
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
        const ExpertCacheStatistics cache_after = model.expert_cache->statistics();
        statistics.expert_cache_hits += cache_after.hits - cache_before.hits;
        statistics.expert_cache_misses += cache_after.misses - cache_before.misses;
        statistics.expert_cache_evictions += cache_after.evictions - cache_before.evictions;
        statistics.expert_cache_bytes_read += cache_after.bytes_read - cache_before.bytes_read;
        statistics.expert_cache_io_worker_count = cache_after.io_worker_count;
        statistics.expert_cache_adaptive_io_workers = cache_after.adaptive_io_workers;
        statistics.expert_cache_io_read_samples += cache_after.io_read_samples - cache_before.io_read_samples;
        statistics.expert_cache_io_read_time_microseconds += cache_after.io_read_time_microseconds
                                                             - cache_before.io_read_time_microseconds;
        statistics.expert_cache_resident_bytes = cache_after.resident_bytes;
    }
    if (model.expert_backend)
    {
        record_expert_backend_delta(
            statistics,
            backend_before,
            model.expert_backend->statistics());
    }
    record_vulkan_execution_delta(statistics, vulkan_before, model.vulkan_context_instance);
    ++statistics.speculative_proposals;
    statistics.speculative_draft_tokens += proposal.token_ids.size();
    statistics.speculative_draft_time_microseconds += elapsed_microseconds(started);
    return proposal;
}

Result<CpuSpeculativeProposal> CpuExecutor::propose_speculative(const CompiledModel& model, int32_t input_id, SessionStatistics& statistics, CpuSessionState& state, uint64_t position_offset,
                                                                const CpuSpeculativeSampler& sampler) const
{
    if (!model.speculative.enabled())
    {
        return Error{
            ErrorCode::UnsupportedModel,
            "the model does not provide a speculative execution plan"};
    }
    Bfloat16BatchedLinearExecutionCounter cpu_bfloat16_execution;
    const ScopedBfloat16BatchedLinearExecutionCounter cpu_bfloat16_scope(
        &cpu_bfloat16_execution);
    if (model.speculative.kind == SpeculativeModelKind::Mtp)
    {
        auto proposal = propose_mtp(
            model,
            input_id,
            statistics,
            state,
            position_offset,
            sampler);
        if (proposal)
        {
            statistics.cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_execution.dispatch_count();
        }
        return proposal;
    }
    if (!sampler)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "DSpark proposal requires a token sampler"};
    }
    auto layer_nodes = collect_speculative_layer_execution_nodes(model);
    if (!layer_nodes)
        return layer_nodes.error();
    if (input_id < 0
        || static_cast<uint32_t>(input_id)
               >= model.descriptor.vocabulary_size
        || state.speculative_layers.size()
               != layer_nodes.value().size())
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
    const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution(model.vulkan_context_instance);
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
    for (size_t layer_index = 0; layer_index < layer_nodes.value().size(); ++layer_index)
    {
        auto executed = execute_speculative_layer(
            model,
            layer_nodes.value()[layer_index],
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
                                        model.descriptor.hyper_connection_epsilon,
                                        model.optimization_flags);
    if (!headed)
        return headed.error();
    CpuBatch head_hidden = std::move(headed).value();
    CpuBatch normalized = rms_norm_batch(head_hidden, model.weights.at(model.speculative.final_norm_weight), model.descriptor.norm_epsilon, model.descriptor.norm_weight_offset, model.optimization_flags);
    CpuBatch base_logits = linear_batch(
        model.weights.at(model.lm_head_weight),
        normalized,
        model.optimization_flags,
        &model.operators.at_weight(model.lm_head_weight));

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
        CpuBatch markov_logits = linear_batch(
            model.weights.at(model.speculative.markov_head_weight),
            markov_embedding,
            model.optimization_flags,
            &model.operators.at_weight(model.speculative.markov_head_weight));
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
        statistics.expert_cache_io_worker_count = cache_after.io_worker_count;
        statistics.expert_cache_adaptive_io_workers = cache_after.adaptive_io_workers;
        statistics.expert_cache_io_read_samples += cache_after.io_read_samples - cache_before.io_read_samples;
        statistics.expert_cache_io_read_time_microseconds += cache_after.io_read_time_microseconds - cache_before.io_read_time_microseconds;
        statistics.expert_cache_resident_bytes = cache_after.resident_bytes;
    }
    if (model.expert_backend)
    {
        record_expert_backend_delta(statistics, backend_before, model.expert_backend->statistics());
    }
    record_vulkan_execution_delta(statistics, vulkan_before, model.vulkan_context_instance);
    statistics.cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_execution.dispatch_count();
    ++statistics.speculative_proposals;
    statistics.speculative_draft_tokens += proposal.token_ids.size();
    statistics.speculative_draft_time_microseconds += elapsed_microseconds(started);
    return proposal;
}

static bool has_unknown_vulkan_attention_state(
    const CpuSessionState& state) noexcept
{
    return std::any_of(
        state.layers.begin(),
        state.layers.end(),
        [](const CpuLayerCache& cache) {
            return cache.vulkan_attention_state_unknown;
        });
}

Result<std::vector<std::vector<float>>> CpuExecutor::execute(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    SessionStatistics& statistics,
    CpuSessionState& state,
    uint64_t position_offset) const
{
    const ScopedExpertBackendForeground expert_backend_foreground(
        model.expert_backend);
    if (has_unknown_vulkan_attention_state(state))
    {
        return Error{
            ErrorCode::InternalError,
            "Session state is unavailable after a failed Vulkan Attention update"};
    }
    const VulkanExecutionSnapshot initial_vulkan_execution = capture_vulkan_execution(model.vulkan_context_instance);
    Bfloat16BatchedLinearExecutionCounter cpu_bfloat16_execution;
    const ScopedBfloat16BatchedLinearExecutionCounter cpu_bfloat16_scope(
        &cpu_bfloat16_execution);
    for (int32_t token_id : input_ids)
    {
        if (token_id < 0 || static_cast<uint32_t>(token_id) >= model.descriptor.vocabulary_size)
            return Error{ErrorCode::InvalidArgument, "token id is outside the model vocabulary"};
    }

    if (statistics.expert_token_counts.size() < model.descriptor.expert_count)
        statistics.expert_token_counts.resize(model.descriptor.expert_count, 0);

    if (state.layers.size() != model.graph.layer_plans.size())
        state.layers.resize(model.graph.layer_plans.size());
    if (state.execution_layers.size() != model.graph.layer_plans.size())
    {
        state.execution_layers.resize(model.graph.layer_plans.size());
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
        const uint32_t speculative_hidden_columns = model.speculative.kind == SpeculativeModelKind::Mtp
                                                        ? model.descriptor.hidden_size
                                                        : model.descriptor.hidden_size
                                                              * static_cast<uint32_t>(
                                                                  model.speculative.target_layer_ids.size());
        state.speculative_main_hidden.reset(
            input_ids.size(),
            speculative_hidden_columns,
            true);
        state.speculative_main_hidden_position = position_offset;
        if (state.speculative_layers.size() != model.speculative.graph.layer_plans.size())
            state.speculative_layers.resize(model.speculative.graph.layer_plans.size());
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
    bool deferred_final_norm = false;
    ExecutionEventRuntime event_runtime(model.graph.events);
    for (const ExecutionBackendRun& backend_run : model.schedule.backend_runs)
    {
        for (uint32_t run_offset = 0;
             run_offset < backend_run.node_count;
             ++run_offset)
        {
            if (backend_run.first_node + run_offset >= model.schedule.node_order.size())
                return Error{ErrorCode::InternalError, "execution backend run exceeds the execution reservation"};
            const ExecutionNodeId node_id = model.schedule.node_order[backend_run.first_node + run_offset];
            if (node_id >= model.graph.nodes.size())
                return Error{ErrorCode::InternalError, "execution schedule references an invalid node"};
            const ExecutionNode* node = &model.graph.nodes[node_id];
            auto waited = event_runtime.wait(node->wait_events);
            if (!waited)
                return waited.error();
            const ExecutionNodeEventGuard event_guard(event_runtime, node->signal_event);

            if (node->type == ExecutionNodeType::TokenEmbedding)
            {
                if (node->weight_inputs.size() != 1)
                    return Error{ErrorCode::InternalError, "token embedding node has an invalid weight binding"};
                const auto embedding_start = std::chrono::steady_clock::now();
                embedding_batch_into(
                    model.weights.at(node->weight_inputs[0]),
                    input_ids,
                    hidden);
                expand_hyper_connections(hidden, model.descriptor.hyper_connection_multiplier, state.expert_scratch.staged_output);
                statistics.embedding_time_microseconds += elapsed_microseconds(embedding_start);
                continue;
            }
            if (node->type == ExecutionNodeType::FinalNorm)
            {
                if (node->weight_inputs.size() != 2)
                    return Error{ErrorCode::InternalError, "final norm node has an invalid weight binding"};
                const CompiledOperator& lm_head_operator = model.operators.at_weight(
                    node->weight_inputs[1]);
                if (model.descriptor.hyper_connection_multiplier == 1
                    && !state.speculative_context_enabled
                    && lm_head_operator.bfloat16
                    && lm_head_operator.bfloat16->has_rms_norm_chain())
                {
                    deferred_final_norm = true;
                    continue;
                }
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
                        model.descriptor.hyper_connection_epsilon,
                        model.optimization_flags);
                    if (!head)
                        return head.error();
                    hidden = std::move(head).value();
                }
                rms_norm_batch_into(
                    hidden,
                    model.weights.at(node->weight_inputs[0]),
                    model.descriptor.norm_epsilon,
                    state.expert_scratch.staged_output,
                    model.descriptor.norm_weight_offset,
                    model.optimization_flags);
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
                if (node->weight_inputs.size() != 2)
                    return Error{ErrorCode::InternalError, "LM head node has an invalid weight binding"};
                const auto lm_head_start = std::chrono::steady_clock::now();
                const auto& lm_head = model.weights.at(node->weight_inputs[0]);
                const CompiledOperator& lm_head_operator = model.operators.at_weight(node->weight_inputs[0]);
                if (deferred_final_norm
                    && try_fused_rms_norm_linear(
                        lm_head_operator,
                        hidden,
                        state.expert_scratch.staged_output))
                {
                    logits = batch_to_vectors(state.expert_scratch.staged_output);
                    deferred_final_norm = false;
                    statistics.lm_head_time_microseconds += elapsed_microseconds(lm_head_start);
                    continue;
                }
                if (deferred_final_norm)
                {
                    CpuBatch normalized;
                    rms_norm_batch_into(
                        hidden,
                        model.weights.at(node->weight_inputs[1]),
                        model.descriptor.norm_epsilon,
                        normalized,
                        model.descriptor.norm_weight_offset,
                        model.optimization_flags);
                    hidden.swap(normalized);
                    deferred_final_norm = false;
                }
                logits = batch_to_vectors(linear_batch(
                    lm_head,
                    hidden,
                    model.optimization_flags,
                    &model.operators.at_weight(node->weight_inputs[0]),
                    node->backend));
                statistics.lm_head_time_microseconds += elapsed_microseconds(lm_head_start);
                continue;
            }
            if (node->layer_plan_index >= model.graph.layer_plans.size())
                return Error{ErrorCode::InternalError, "execution node layer is out of range"};

            const CompiledLayerPlan& layer = model.graph.layer_plans[node->layer_plan_index];
            if (layer.layer_id >= layer_states.size())
                return Error{ErrorCode::InternalError, "execution layer id is out of range"};
            LayerGraphState& layer_state = layer_states[layer.layer_id];
            const MoeBlockPlan& moe = layer.moe;
            if (node->type == ExecutionNodeType::Attention)
            {
                const auto attention_start = std::chrono::steady_clock::now();
                if (has_flag(layer.attention.flags, AttentionBlockGatedDeltaNet))
                {
                    auto gated_delta = execute_gated_delta_net_into(
                        model.weights,
                        model.operators,
                        layer.attention,
                        node->backend,
                        model.descriptor.norm_epsilon,
                        state.layers[layer.layer_id],
                        state.gated_delta_scratch,
                        hidden,
                        state.gated_delta_scratch.output,
                        model.optimization_flags);
                    if (!gated_delta)
                        return gated_delta.error();
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
                            model.descriptor.hyper_connection_epsilon,
                            model.optimization_flags);
                        if (!mixed)
                            return mixed.error();
                        hyper_mix = std::move(mixed).value();
                        attention_input = &hyper_mix.reduced;
                    }
                    auto output = execute_latent_attention(
                        model.weights,
                        model.operators,
                        layer.attention,
                        node->backend,
                        model.descriptor.norm_epsilon,
                        position_offset,
                        state.layers[layer.layer_id],
                        *attention_input,
                        model.optimization_flags);
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
                    auto attention = execute_attention_block_into(
                        model.weights,
                        model.operators,
                        layer.attention,
                        node->backend,
                        model.descriptor.norm_epsilon,
                        model.descriptor.kv_cache_dtype,
                        position_offset,
                        state.layers[layer.layer_id],
                        state.attention_scratch,
                        hidden,
                        state.attention_scratch.output,
                        model.optimization_flags);
                    if (!attention)
                        return attention.error();
                    hidden.swap(state.attention_scratch.output);
                }
                statistics.attention_time_microseconds += elapsed_microseconds(attention_start);
                continue;
            }
            if (node->type == ExecutionNodeType::Router)
            {
                auto completed_prediction = complete_router_prediction(
                    model,
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
                        model.descriptor.hyper_connection_epsilon,
                        model.optimization_flags);
                    if (!mixed)
                        return mixed.error();
                    layer_state.ffn_hyper_mix = std::move(mixed).value();
                    rms_norm_batch_into(layer_state.ffn_hyper_mix.reduced, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset, model.optimization_flags);
                }
                else
                {
                    rms_norm_batch_into(hidden, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset, model.optimization_flags);
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
                linear_batch_into(
                    model.weights.at(moe.router_weight),
                    layer_state.normalized,
                    layer_state.router_logits,
                    model.optimization_flags,
                    &model.operators.at_weight(moe.router_weight));
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
                const auto expert_engine_start = std::chrono::steady_clock::now();
                auto executed = run_moe(
                    model,
                    moe,
                    layer_state,
                    statistics,
                    state.expert_scratch,
                    layer.layer_id,
                    node->backend,
                    has_flag(node->flags, ExecutionNodeCpuPrefetch));
                statistics.expert_engine_time_microseconds += elapsed_microseconds(expert_engine_start);
                if (!executed)
                    return executed.error();
                continue;
            }
            if (node->type == ExecutionNodeType::SharedExpertGroup)
            {
                if (!layer_state.experts_executed || !moe.has_shared_expert)
                    return Error{ErrorCode::InternalError, "Shared Expert executed before routed Expert group"};
                const auto shared_start = std::chrono::steady_clock::now();
                ExpertExecutionMetrics shared_metrics;
                layer_state.shared_expert_output = run_shared_expert(
                    model,
                    moe,
                    layer_state.normalized,
                    shared_metrics,
                    model.optimization_flags);
                statistics.expert_compute_time_microseconds += elapsed_microseconds(shared_start);
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
                    return Error{ErrorCode::InternalError, "Combine executed before Shared Expert group"};
                CpuBatch& moe_output = layer_state.normalized;
                const bool has_backend_aggregation = initialize_backend_aggregated_output(
                    state.expert_scratch,
                    layer_state.active_experts,
                    hidden.rows(),
                    model.descriptor.hidden_size,
                    moe_output);
                for (size_t active_index = 0; active_index < layer_state.active_experts.size(); ++active_index)
                {
                    const ActiveExpertExecution& active = layer_state.active_experts[active_index];
                    if (has_backend_aggregation
                        && active_index < state.expert_scratch.backend_aggregated.size()
                        && state.expert_scratch.backend_aggregated[active_index] != 0)
                    {
                        continue;
                    }
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
        statistics.expert_cache_io_worker_count = after.io_worker_count;
        statistics.expert_cache_adaptive_io_workers = after.adaptive_io_workers;
        statistics.expert_cache_io_read_samples += after.io_read_samples - execution_cache_before.io_read_samples;
        statistics.expert_cache_io_read_time_microseconds += after.io_read_time_microseconds - execution_cache_before.io_read_time_microseconds;
        statistics.expert_cache_resident_bytes = after.resident_bytes;
        record_expert_victim_cache_delta(statistics, execution_cache_before.victim, after.victim);
    }
    if (model.expert_backend)
    {
        record_expert_backend_delta(statistics, expert_backend_before, model.expert_backend->statistics());
    }

    if (logits.empty())
        return Error{ErrorCode::InternalError, "execution graph did not produce logits"};
    record_vulkan_execution_delta(
        statistics,
        initial_vulkan_execution,
        model.vulkan_context_instance);
    statistics.cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_execution.dispatch_count();
    auto residency = state.memory_manager.record_execution(
        model.graph,
        model.schedule);
    if (!residency)
        return residency.error();
    return logits;
}

Result<std::vector<std::vector<float>>> CpuExecutor::execute_decode_batch(const CompiledModel& model, std::span<const CpuDecodeBatchEntry> entries, CpuDecodeBatchMetrics& metrics) const
{
    const ScopedExpertBackendForeground expert_backend_foreground(
        model.expert_backend);
    if (entries.empty())
        return Error{ErrorCode::InvalidArgument, "decode batch cannot be empty"};

    const size_t session_count = entries.size();
    Bfloat16BatchedLinearExecutionCounter cpu_bfloat16_execution;
    const ScopedBfloat16BatchedLinearExecutionCounter cpu_bfloat16_scope(
        &cpu_bfloat16_execution);
    for (const CpuDecodeBatchEntry& entry : entries)
    {
        if (!entry.state || has_unknown_vulkan_attention_state(*entry.state))
        {
            return Error{
                ErrorCode::InternalError,
                "staged Session state is unavailable after a failed Vulkan Attention update"};
        }
    }
    const uint32_t hyper_multiplier = model.descriptor.hyper_connection_multiplier;
    const uint32_t hyper_iterations = model.descriptor.hyper_connection_iterations;
    const float hyper_epsilon = model.descriptor.hyper_connection_epsilon;
    std::vector<CpuBatch> hidden(session_count);
    std::vector<std::vector<float>> logits(session_count);
    CpuExpertExecutionScratch& batch_scratch = entries.front().state->expert_scratch;
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
        if (entry.state->layers.size() != model.graph.layer_plans.size())
            entry.state->layers.resize(model.graph.layer_plans.size());
        if (entry.state->execution_layers.size() != model.graph.layer_plans.size())
        {
            entry.state->execution_layers.resize(model.graph.layer_plans.size());
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
        const uint32_t speculative_hidden_columns = model.speculative.kind == SpeculativeModelKind::Mtp
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
            if (entry.state->speculative_layers.size() != model.speculative.graph.layer_plans.size())
            {
                entry.state->speculative_layers.resize(model.speculative.graph.layer_plans.size());
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
    bool speculative_context_enabled = false;
    for (const CpuDecodeBatchEntry& entry : entries)
    {
        speculative_context_enabled = speculative_context_enabled
                                      || entry.state->speculative_context_enabled;
    }
    bool deferred_final_norm = false;
    ExecutionEventRuntime event_runtime(model.graph.events);
    for (const ExecutionBackendRun& backend_run : model.schedule.backend_runs)
    {
        for (uint32_t run_offset = 0;
             run_offset < backend_run.node_count;
             ++run_offset)
        {
            if (backend_run.first_node + run_offset >= model.schedule.node_order.size())
            {
                return Error{ErrorCode::InternalError, "execution backend run exceeds the execution reservation"};
            }
            const ExecutionNodeId node_id = model.schedule.node_order[backend_run.first_node + run_offset];
            if (node_id >= model.graph.nodes.size())
            {
                return Error{ErrorCode::InternalError, "execution schedule references an invalid node"};
            }
            const ExecutionNode* node = &model.graph.nodes[node_id];
            auto waited = event_runtime.wait(node->wait_events);
            if (!waited)
                return waited.error();
            const ExecutionNodeEventGuard event_guard(event_runtime, node->signal_event);

            if (node->type == ExecutionNodeType::TokenEmbedding)
            {
                if (node->weight_inputs.size() != 1)
                    return Error{ErrorCode::InternalError, "token embedding node has an invalid weight binding"};
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                std::vector<int32_t>& input_ids = scratch.staged_input_ids;
                input_ids.resize(session_count);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    input_ids[session_index] = entries[session_index].input_id;
                }
                embedding_batch_into(
                    model.weights.at(node->weight_inputs[0]),
                    input_ids,
                    scratch.staged_output);
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
                if (node->weight_inputs.size() != 2)
                    return Error{ErrorCode::InternalError, "final norm node has an invalid weight binding"};
                const CompiledOperator& lm_head_operator = model.operators.at_weight(
                    node->weight_inputs[1]);
                if (hyper_multiplier == 1
                    && !speculative_context_enabled
                    && lm_head_operator.bfloat16
                    && lm_head_operator.bfloat16->has_rms_norm_chain())
                {
                    deferred_final_norm = true;
                    continue;
                }
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
                                                      hyper_multiplier, model.descriptor.norm_epsilon, hyper_epsilon,
                                                      model.optimization_flags);
                    if (!head)
                        return head.error();
                    split_rows(head.value(), hidden);
                }
                CpuBatch& merged = scratch.staged_merged;
                if (!merge_rows_into(hidden, merged))
                {
                    return Error{ErrorCode::InternalError, "cannot merge staged hidden rows"};
                }
                rms_norm_batch_into(merged, model.weights.at(node->weight_inputs[0]), model.descriptor.norm_epsilon, scratch.staged_output,
                                    model.descriptor.norm_weight_offset, model.optimization_flags);
                split_rows(scratch.staged_output, hidden);
                if (model.speculative.kind == SpeculativeModelKind::Mtp)
                {
                    for (size_t session_index = 0;
                         session_index < session_count;
                         ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
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
                if (node->weight_inputs.size() != 2)
                    return Error{ErrorCode::InternalError, "LM head node has an invalid weight binding"};
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                CpuBatch& merged = scratch.staged_merged;
                if (!merge_rows_into(hidden, merged))
                {
                    return Error{ErrorCode::InternalError, "cannot merge staged LM head rows"};
                }
                const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution(model.vulkan_context_instance);
                const auto& lm_head = model.weights.at(node->weight_inputs[0]);
                const CompiledOperator& lm_head_operator = model.operators.at_weight(node->weight_inputs[0]);
                if (deferred_final_norm
                    && try_fused_rms_norm_linear(
                        lm_head_operator,
                        merged,
                        scratch.staged_output))
                {
                    deferred_final_norm = false;
                }
                else
                {
                    if (deferred_final_norm)
                    {
                        CpuBatch normalized;
                        rms_norm_batch_into(
                            merged,
                            model.weights.at(node->weight_inputs[1]),
                            model.descriptor.norm_epsilon,
                            normalized,
                            model.descriptor.norm_weight_offset,
                            model.optimization_flags);
                        linear_batch_into(
                            lm_head,
                            normalized,
                            scratch.staged_output,
                            model.optimization_flags,
                            &model.operators.at_weight(node->weight_inputs[0]),
                            node->backend);
                        deferred_final_norm = false;
                    }
                    else
                    {
                        linear_batch_into(
                            lm_head,
                            merged,
                            scratch.staged_output,
                            model.optimization_flags,
                            &model.operators.at_weight(node->weight_inputs[0]),
                            node->backend);
                    }
                }
                const std::vector<std::vector<float>> merged_logits = batch_to_vectors(scratch.staged_output);
                const uint64_t elapsed = elapsed_microseconds(start);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    logits[session_index] = merged_logits[session_index];
                    entries[session_index].statistics->lm_head_time_microseconds += elapsed;
                }
                for (const CpuDecodeBatchEntry& entry : entries)
                    record_vulkan_execution_delta(*entry.statistics, vulkan_before, model.vulkan_context_instance);
                continue;
            }
            if (node->layer_plan_index >= model.graph.layer_plans.size())
            {
                return Error{ErrorCode::InternalError, "execution node layer is out of range"};
            }

            const CompiledLayerPlan& layer = model.graph.layer_plans[node->layer_plan_index];
            const MoeBlockPlan& moe = layer.moe;
            if (node->type == ExecutionNodeType::Attention)
            {
                if (has_flag(layer.attention.flags, AttentionBlockGatedDeltaNet))
                {
                    std::vector<CpuGatedDeltaBatchEntry>& gated_delta_entries = batch_scratch.gated_delta_entries;
                    gated_delta_entries.resize(session_count);
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        gated_delta_entries[session_index] = {
                            &hidden[session_index],
                            &state.gated_delta_scratch,
                            &state.layers[layer.layer_id],
                            &state.gated_delta_scratch.output};
                    }
                    const auto start = std::chrono::steady_clock::now();
                    const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution(model.vulkan_context_instance);
                    if (!execute_gated_delta_net_batch_into(
                            model.weights,
                            model.operators,
                            layer.attention,
                            node->backend,
                            model.descriptor.norm_epsilon,
                            gated_delta_entries,
                            batch_scratch.gated_delta_batch,
                            model.optimization_flags))
                    {
                        return Error{
                            ErrorCode::InternalError,
                            "gated delta batch execution failed"};
                    }
                    const uint64_t elapsed = elapsed_microseconds(start);
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        hidden[session_index].swap(state.gated_delta_scratch.output);
                        SessionStatistics& statistics = *entries[session_index].statistics;
                        statistics.attention_time_microseconds += elapsed;
                        record_vulkan_execution_delta(statistics, vulkan_before, model.vulkan_context_instance);
                    }
                }
                else if (has_flag(layer.attention.flags, AttentionBlockLatent))
                {
                    const auto start = std::chrono::steady_clock::now();
                    const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution(model.vulkan_context_instance);
                    std::vector<uint64_t>& positions = batch_scratch.staged_attention_positions;
                    std::vector<CpuLayerCache*>& caches = batch_scratch.staged_attention_caches;
                    positions.resize(session_count);
                    caches.resize(session_count);
                    CpuExpertExecutionScratch& scratch = batch_scratch;
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
                                                          model.weights.at(layer.hyper_connection.attention_base), hyper_multiplier, hyper_iterations,
                                                          model.descriptor.norm_epsilon, hyper_epsilon, model.optimization_flags);
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
                    auto merged_output = execute_latent_attention_batch(model.weights, model.operators, layer.attention, node->backend, model.descriptor.norm_epsilon, positions, caches,
                                                                        *attention_input, model.optimization_flags);
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
                        std::vector<CpuBatch>& attention_outputs = batch_scratch.staged_batches;
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
                        record_vulkan_execution_delta(*entries[session_index].statistics, vulkan_before, model.vulkan_context_instance);
                    }
                }
                else
                {
                    std::vector<CpuAttentionBatchEntry>& attention_entries = batch_scratch.attention_batch_entries;
                    attention_entries.resize(session_count);
                    for (size_t session_index = 0;
                         session_index < session_count;
                         ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        attention_entries[session_index] = {
                            entries[session_index].position_offset,
                            &state.layers[layer.layer_id],
                            &state.attention_scratch,
                            &hidden[session_index],
                            &state.attention_scratch.output};
                    }
                    const auto batch_start = std::chrono::steady_clock::now();
                    const VulkanExecutionSnapshot batch_vulkan_before = capture_vulkan_execution(model.vulkan_context_instance);
                    auto batched = execute_attention_block_batch_into(
                        model.operators,
                        layer.attention,
                        node->backend,
                        attention_entries,
                        model.optimization_flags);
                    if (!batched)
                        return batched.error();
                    if (batched.value())
                    {
                        const uint64_t elapsed = elapsed_microseconds(batch_start);
                        ++metrics.vulkan_attention_batch_submissions;
                        metrics.vulkan_attention_batch_rows += session_count;
                        metrics.vulkan_attention_batch_avoided_submissions += session_count - 1;
                        for (size_t session_index = 0;
                             session_index < session_count;
                             ++session_index)
                        {
                            CpuSessionState& state = *entries[session_index].state;
                            hidden[session_index].swap(
                                state.attention_scratch.output);
                            SessionStatistics& statistics = *entries[session_index].statistics;
                            statistics.attention_time_microseconds += elapsed;
                            record_vulkan_execution_delta(
                                statistics,
                                batch_vulkan_before, model.vulkan_context_instance);
                        }
                    }
                    else
                    {
                        for (size_t session_index = 0;
                             session_index < session_count;
                             ++session_index)
                        {
                            CpuSessionState& state = *entries[session_index].state;
                            const auto start = std::chrono::steady_clock::now();
                            const VulkanExecutionSnapshot vulkan_before = capture_vulkan_execution(model.vulkan_context_instance);
                            auto attention = execute_attention_block_into(
                                model.weights,
                                model.operators,
                                layer.attention,
                                node->backend,
                                model.descriptor.norm_epsilon,
                                model.descriptor.kv_cache_dtype,
                                entries[session_index].position_offset,
                                state.layers[layer.layer_id],
                                state.attention_scratch,
                                hidden[session_index],
                                state.attention_scratch.output,
                                model.optimization_flags);
                            if (!attention)
                            {
                                for (size_t affected_index = 0;
                                     affected_index < session_count;
                                     ++affected_index)
                                {
                                    CpuLayerCache& affected_cache = entries[affected_index].state->layers[layer.layer_id];
                                    affected_cache.vulkan_attention_state_unknown = true;
                                }
                                return attention.error();
                            }
                            hidden[session_index].swap(
                                state.attention_scratch.output);
                            SessionStatistics& statistics = *entries[session_index].statistics;
                            statistics.attention_time_microseconds += elapsed_microseconds(start);
                            record_vulkan_execution_delta(
                                statistics,
                                vulkan_before, model.vulkan_context_instance);
                        }
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
                                                      model.weights.at(layer.hyper_connection.ffn_base), hyper_multiplier, hyper_iterations,
                                                      model.descriptor.norm_epsilon, hyper_epsilon, model.optimization_flags);
                    if (!mixed)
                        return mixed.error();
                    CpuHyperConnectionMix merged_mix = std::move(mixed).value();
                    rms_norm_batch_into(merged_mix.reduced, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, merged_hidden,
                                        model.descriptor.norm_weight_offset, model.optimization_flags);
                    std::vector<CpuHyperConnectionMix>& mixes = batch_scratch.staged_hyper_mixes;
                    split_hyper_mix(merged_mix, mixes);
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                        layer_state.ffn_hyper_mix = std::move(mixes[session_index]);
                    }
                }
                else
                {
                    rms_norm_batch_into(merged_hyper, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, merged_hidden,
                                        model.descriptor.norm_weight_offset, model.optimization_flags);
                }
                std::vector<CpuBatch>& normalized = batch_scratch.staged_batches;
                split_rows(merged_hidden, normalized);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    layer_state.normalized = std::move(normalized[session_index]);
                }
                CpuBatch& merged_logits = scratch.staged_router_logits;
                linear_batch_into(
                    model.weights.at(moe.router_weight),
                    merged_hidden,
                    merged_logits,
                    model.optimization_flags,
                    &model.operators.at_weight(moe.router_weight));
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
                    std::vector<uint32_t>& explicit_expert_ids = batch_scratch.staged_expert_ids;
                    explicit_expert_ids.clear();
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
                LayerGraphState combined;
                combined.normalized.reset(session_count, model.descriptor.hidden_size, false);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    const LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    std::copy_n(layer_state.normalized.row(0), combined.normalized.columns(), combined.normalized.row(session_index));
                }
                const size_t missing = std::numeric_limits<size_t>::max();
                std::vector<size_t>& combined_by_expert = batch_scratch.combined_by_expert;
                combined_by_expert.assign(moe.experts.size(), missing);
                std::vector<std::vector<CpuDecodeRouteOrigin>>& origins = batch_scratch.staged_route_origins;
                origins.clear();
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
                            origins[combined_index].push_back(
                                {session_index, active_index, route_index});
                        }
                    }
                }

                SessionStatistics aggregate_statistics;
                aggregate_statistics.expert_token_counts.resize(model.descriptor.expert_count, 0);
                const auto engine_start = std::chrono::steady_clock::now();
                auto executed = run_moe(
                    model,
                    moe,
                    combined,
                    aggregate_statistics,
                    entries.front().state->expert_scratch,
                    layer.layer_id,
                    node->backend,
                    has_flag(node->flags, ExecutionNodeCpuPrefetch));
                const uint64_t engine_elapsed = elapsed_microseconds(engine_start);
                if (!executed)
                    return executed.error();

                // Snapshot aggregated output before reusing session zero's scratch.
                const CpuExpertExecutionScratch& combined_scratch = entries.front().state->expert_scratch;
                std::vector<uint8_t>& combined_backend_aggregated = batch_scratch.combined_backend_aggregated;
                combined_backend_aggregated = combined_scratch.backend_aggregated;
                const bool has_combined_backend_aggregation = std::any_of(
                                                                  combined_backend_aggregated.begin(),
                                                                  combined_backend_aggregated.end(),
                                                                  [](uint8_t value) { return value != 0; })
                                                              && combined_scratch.backend_aggregated_output.rows()
                                                                     == session_count
                                                              && combined_scratch.backend_aggregated_output.columns()
                                                                     == model.descriptor.hidden_size;
                CpuBatch& combined_backend_aggregated_output = batch_scratch.combined_backend_aggregated_output;
                combined_backend_aggregated_output.clear();
                if (has_combined_backend_aggregation)
                {
                    combined_backend_aggregated_output = combined_scratch.backend_aggregated_output;
                }
                for (size_t session_index = 0;
                     session_index < session_count;
                     ++session_index)
                {
                    CpuExpertExecutionScratch& session_scratch = entries[session_index].state->expert_scratch;
                    const LayerGraphState& session_layer = entries[session_index].state->execution_layers[layer.layer_id];
                    session_scratch.backend_aggregated.assign(
                        session_layer.active_experts.size(),
                        0);
                    if (has_combined_backend_aggregation)
                    {
                        session_scratch.backend_aggregated_output.reset(
                            1,
                            model.descriptor.hidden_size,
                            false);
                        std::copy_n(
                            combined_backend_aggregated_output.row(
                                session_index),
                            model.descriptor.hidden_size,
                            session_scratch.backend_aggregated_output.row(0));
                    }
                    else
                    {
                        session_scratch.backend_aggregated_output.clear();
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
                    const bool backend_aggregated = has_combined_backend_aggregation
                                                    && combined_index
                                                           < combined_backend_aggregated.size()
                                                    && combined_backend_aggregated[combined_index] != 0;
                    for (size_t route_index = 0; route_index < origins[combined_index].size(); ++route_index)
                    {
                        const CpuDecodeRouteOrigin& origin = origins[combined_index][route_index];
                        ActiveExpertExecution& destination = entries[origin.session_index].state->execution_layers[layer.layer_id].active_experts[origin.active_index];
                        if (backend_aggregated)
                        {
                            CpuExpertExecutionScratch& origin_scratch = entries[origin.session_index].state->expert_scratch;
                            origin_scratch.backend_aggregated[origin.active_index] = 1;
                            continue;
                        }
                        std::copy_n(
                            source.output.row(route_index),
                            model.descriptor.hidden_size,
                            destination.output.row(origin.route_index));
                    }
                }
                continue;
            }
            if (node->type == ExecutionNodeType::SharedExpertGroup)
            {
                if (!moe.has_shared_expert)
                    return Error{ErrorCode::InternalError, "Shared Expert graph node has no shared Expert plan"};
                CpuBatch shared_input;
                shared_input.reset(session_count, model.descriptor.hidden_size, false);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    const CpuBatch& normalized = entries[session_index].state->execution_layers[layer.layer_id].normalized;
                    if (normalized.rows() != 1 || normalized.columns() != model.descriptor.hidden_size)
                        return Error{ErrorCode::InternalError, "Shared Expert input has an invalid shape"};
                    std::copy_n(normalized.row(0), normalized.columns(), shared_input.row(session_index));
                }
                const auto shared_start = std::chrono::steady_clock::now();
                ExpertExecutionMetrics shared_metrics;
                const CpuBatch shared_output = run_shared_expert(
                    model,
                    moe,
                    shared_input,
                    shared_metrics,
                    model.optimization_flags);
                const uint64_t shared_elapsed = elapsed_microseconds(shared_start);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    layer_state.shared_expert_output.reset(1, shared_output.columns(), false);
                    std::copy_n(shared_output.row(session_index), shared_output.columns(), layer_state.shared_expert_output.row(0));
                    entries[session_index].statistics->expert_compute_time_microseconds += shared_elapsed;
                }
                continue;
            }
            if (node->type == ExecutionNodeType::Combine)
            {
                const auto combine_start = std::chrono::steady_clock::now();
                std::vector<CpuBatch>& moe_outputs = batch_scratch.staged_batches;
                moe_outputs.resize(session_count);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    if (!layer_state.experts_executed)
                    {
                        return Error{ErrorCode::InternalError, "Combine executed before its Expert wave"};
                    }
                    if (moe.has_shared_expert && layer_state.shared_expert_output.rows() == 0)
                        return Error{ErrorCode::InternalError, "Combine executed before Shared Expert group"};
                    CpuBatch& moe_output = layer_state.normalized;
                    const CpuExpertExecutionScratch& expert_scratch = entries[session_index].state->expert_scratch;
                    const bool has_backend_aggregation = initialize_backend_aggregated_output(
                        expert_scratch,
                        layer_state.active_experts,
                        1,
                        model.descriptor.hidden_size,
                        moe_output);
                    for (size_t active_index = 0; active_index < layer_state.active_experts.size(); ++active_index)
                    {
                        const ActiveExpertExecution& active = layer_state.active_experts[active_index];
                        if (has_backend_aggregation
                            && active_index < expert_scratch.backend_aggregated.size()
                            && expert_scratch.backend_aggregated[active_index] != 0)
                        {
                            continue;
                        }
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
                    std::vector<CpuHyperConnectionMix>& mixes = batch_scratch.staged_hyper_mixes;
                    mixes.resize(session_count);
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
            statistics.expert_cache_io_worker_count = cache_after.io_worker_count;
            statistics.expert_cache_adaptive_io_workers = cache_after.adaptive_io_workers;
            statistics.expert_cache_io_read_samples += cache_after.io_read_samples - cache_before.io_read_samples;
            statistics.expert_cache_io_read_time_microseconds += cache_after.io_read_time_microseconds - cache_before.io_read_time_microseconds;
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
        auto residency = entries[session_index].state->memory_manager.record_execution(
            model.graph,
            model.schedule);
        if (!residency)
            return residency.error();
    }
    const uint64_t cpu_bfloat16_dispatches = cpu_bfloat16_execution.dispatch_count();
    for (const CpuDecodeBatchEntry& entry : entries)
    {
        entry.statistics->cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_dispatches;
    }
    return logits;
}

} // namespace moe
} // namespace ncnn
