#include "metrics.h"

#include "backends/ncnn/vulkancontext.h"
#include "executor.h"
#include "expertbackend.h"
#include "ncnn/moe/session.h"
#include "storage/expertcache_victim.h"
#include "storage/expertcache.h"
#include "graph/layerplan.h"
#include "graph/compiledmodel.h"

#include <chrono>
#include <limits>

namespace ncnn {
namespace moe {

void record_model_resource_delta(
    const CompiledModel& model,
    SessionStatistics& statistics,
    const ExpertCacheStatistics& execution_cache_before,
    const ExpertBackendStatistics& expert_backend_before)
{
    if (model.expert_cache)
    {
        const ExpertCacheStatistics after = model.expert_cache->statistics();
        record_expert_cache_delta(statistics, execution_cache_before, after);
        statistics.expert_cache_queued_reads += after.queued_reads - execution_cache_before.queued_reads;
        statistics.expert_cache_speculative_reads += after.speculative_reads - execution_cache_before.speculative_reads;
        statistics.expert_cache_cancelled_speculative_reads += after.cancelled_speculative_reads - execution_cache_before.cancelled_speculative_reads;
        statistics.expert_cache_dropped_speculative_admissions += after.dropped_speculative_admissions - execution_cache_before.dropped_speculative_admissions;
        statistics.expert_cache_unused_speculative_reads += after.unused_speculative_reads - execution_cache_before.unused_speculative_reads;
        statistics.expert_cache_short_term_reloads += after.short_term_reloads - execution_cache_before.short_term_reloads;
        statistics.expert_cache_arc_recent_size = after.arc_recent_size;
        statistics.expert_cache_arc_frequent_size = after.arc_frequent_size;
        statistics.expert_cache_arc_recent_target_size = after.arc_recent_target_size;
        statistics.expert_cache_arc_recent_ghost_size = after.arc_recent_ghost_size;
        statistics.expert_cache_arc_frequent_ghost_size = after.arc_frequent_ghost_size;
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
        record_expert_victim_cache_delta(statistics, execution_cache_before.victim, after.victim);
    }
    if (model.expert_backend)
    {
        record_expert_backend_delta(statistics, expert_backend_before, model.expert_backend->statistics());
    }
}

void record_batch_resource_delta(
    const CompiledModel& model,
    std::span<const CpuDecodeBatchEntry> entries,
    const ExpertCacheStatistics& cache_before,
    const ExpertBackendStatistics& backend_before)
{
    // Each participating Session retains the logical batch resource delta.
    if (model.expert_cache)
    {
        const ExpertCacheStatistics cache_after = model.expert_cache->statistics();
        for (const CpuDecodeBatchEntry& entry : entries)
        {
            SessionStatistics& statistics = *entry.statistics;
            record_expert_cache_delta(statistics, cache_before, cache_after);
            statistics.expert_cache_speculative_reads += cache_after.speculative_reads - cache_before.speculative_reads;
            statistics.expert_cache_cancelled_speculative_reads += cache_after.cancelled_speculative_reads - cache_before.cancelled_speculative_reads;
            statistics.expert_cache_dropped_speculative_admissions += cache_after.dropped_speculative_admissions - cache_before.dropped_speculative_admissions;
            statistics.expert_cache_unused_speculative_reads += cache_after.unused_speculative_reads - cache_before.unused_speculative_reads;
            statistics.expert_cache_short_term_reloads += cache_after.short_term_reloads - cache_before.short_term_reloads;
            statistics.expert_cache_coalesced_read_batches += cache_after.coalesced_read_batches - cache_before.coalesced_read_batches;
            statistics.expert_cache_coalesced_experts += cache_after.coalesced_experts - cache_before.coalesced_experts;
            statistics.expert_cache_coalesced_read_ranges_saved += cache_after.coalesced_read_ranges_saved - cache_before.coalesced_read_ranges_saved;
            statistics.expert_cache_arc_recent_size = cache_after.arc_recent_size;
            statistics.expert_cache_arc_frequent_size = cache_after.arc_frequent_size;
            statistics.expert_cache_arc_recent_target_size = cache_after.arc_recent_target_size;
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

void record_expert_weight_demand(const ExpertPlan& expert, size_t route_count, SessionStatistics& statistics) noexcept
{
    add_saturating(expert.weight_size, statistics.expert_batch_weight_bytes);
    add_saturating(saturating_weight_product(expert.weight_size, route_count), statistics.expert_route_weight_bytes);
}

void record_expert_cache_delta(
    SessionStatistics& statistics,
    const ExpertCacheStatistics& before,
    const ExpertCacheStatistics& after)
{
    statistics.expert_cache_hits += after.hits - before.hits;
    statistics.expert_cache_misses += after.misses - before.misses;
    statistics.expert_cache_evictions += after.evictions - before.evictions;
    statistics.expert_cache_bytes_read += after.bytes_read - before.bytes_read;
    statistics.expert_cache_num_io_threads = after.num_io_threads;
    statistics.expert_cache_num_active_io_threads = after.num_active_io_threads;
    statistics.expert_cache_io_read_samples += after.io_read_samples - before.io_read_samples;
    statistics.expert_cache_io_read_time_microseconds += after.io_read_time_microseconds
                                                         - before.io_read_time_microseconds;
    statistics.expert_cache_resident_size = after.resident_size;
}

void record_vulkan_execution_delta(
    SessionStatistics& statistics,
    const NcnnVulkanExecutionSnapshot& before,
    const NcnnVulkanContextInstancePtr& context_instance)
{
    const NcnnVulkanExecutionSnapshot after_snapshot = get_vulkan_execution_snapshot(context_instance);
    const NcnnVulkanRuntimeCounters& after = after_snapshot.counters;
    statistics.vulkan_linear_dispatches += after_snapshot.dispatches - before.dispatches;
    statistics.vulkan_attention_blocks += after_snapshot.attention_blocks - before.attention_blocks;
    statistics.vulkan_compute_submissions += after.compute_submissions - before.counters.compute_submissions;
    statistics.vulkan_submit_wait_time_microseconds += after.submit_wait_time_microseconds
                                                       - before.counters.submit_wait_time_microseconds;
    statistics.vulkan_batch_uploads += after.batch_uploads - before.counters.batch_uploads;
    statistics.vulkan_batch_downloads += after.batch_downloads - before.counters.batch_downloads;
    statistics.vulkan_auxiliary_uploads += after.auxiliary_uploads - before.counters.auxiliary_uploads;
    statistics.vulkan_auxiliary_upload_bytes += after.auxiliary_upload_bytes - before.counters.auxiliary_upload_bytes;
    statistics.vulkan_staging_slot_resizes += after.staging_slot_resizes - before.counters.staging_slot_resizes;
    statistics.vulkan_staging_slot_reuses += after.staging_slot_reuses - before.counters.staging_slot_reuses;
    statistics.vulkan_staging_slot_acquisitions += after.staging_slot_acquisitions - before.counters.staging_slot_acquisitions;
    statistics.vulkan_staging_slot_contentions += after.staging_slot_contentions - before.counters.staging_slot_contentions;
    statistics.vulkan_command_buffer_reuses += after.command_buffer_reuses - before.counters.command_buffer_reuses;
    statistics.vulkan_command_graph_submissions += after.command_graph_submissions
                                                   - before.counters.command_graph_submissions;
    statistics.vulkan_command_graph_operations += after.command_graph_operations
                                                  - before.counters.command_graph_operations;
    statistics.vulkan_direct_host_input_bindings += after.direct_host_input_bindings - before.counters.direct_host_input_bindings;
    statistics.vulkan_direct_host_output_bindings += after.direct_host_output_bindings - before.counters.direct_host_output_bindings;
    statistics.vulkan_attention_qkv_rope_fusions += after.attention_qkv_rope_fusions - before.counters.attention_qkv_rope_fusions;
    statistics.vulkan_attention_device_rope_fusions += after.attention_device_rope_fusions - before.counters.attention_device_rope_fusions;
    statistics.vulkan_attention_qkv_ring_fusions += after.attention_qkv_ring_fusions - before.counters.attention_qkv_ring_fusions;
    statistics.vulkan_attention_qkv_rope_pipeline_failures += after.attention_qkv_rope_pipeline_failures - before.counters.attention_qkv_rope_pipeline_failures;
    statistics.vulkan_attention_qkv_rope_shape_failures += after.attention_qkv_rope_shape_failures - before.counters.attention_qkv_rope_shape_failures;
    statistics.vulkan_attention_qkv_rope_source_failures += after.attention_qkv_rope_source_failures - before.counters.attention_qkv_rope_source_failures;
    statistics.vulkan_attention_qkv_rope_norm_failures += after.attention_qkv_rope_norm_failures - before.counters.attention_qkv_rope_norm_failures;
    statistics.vulkan_attention_qkv_rope_ring_failures += after.attention_qkv_rope_ring_failures - before.counters.attention_qkv_rope_ring_failures;
    statistics.vulkan_attention_qkv_rope_allocation_failures += after.attention_qkv_rope_allocation_failures - before.counters.attention_qkv_rope_allocation_failures;
    statistics.vulkan_attention_precondition_failures += after.attention_precondition_failures - before.counters.attention_precondition_failures;
    statistics.vulkan_attention_staging_failures += after.attention_staging_failures - before.counters.attention_staging_failures;
    statistics.vulkan_attention_norm_failures += after.attention_norm_failures - before.counters.attention_norm_failures;
    statistics.vulkan_attention_qkv_failures += after.attention_qkv_failures - before.counters.attention_qkv_failures;
    statistics.vulkan_attention_cache_failures += after.attention_cache_failures - before.counters.attention_cache_failures;
    statistics.vulkan_attention_sdpa_failures += after.attention_sdpa_failures - before.counters.attention_sdpa_failures;
    statistics.vulkan_attention_projection_failures += after.attention_projection_failures - before.counters.attention_projection_failures;
    statistics.vulkan_attention_output_failures += after.attention_output_failures - before.counters.attention_output_failures;
    statistics.vulkan_attention_submit_failures += after.attention_submit_failures - before.counters.attention_submit_failures;
    statistics.vulkan_attention_decode_sdpa_fusions += after.attention_decode_sdpa_fusions - before.counters.attention_decode_sdpa_fusions;
    statistics.vulkan_attention_cache_materializations += after.attention_cache_materializations
                                                          - before.counters.attention_cache_materializations;
    statistics.vulkan_attention_cpu_fallbacks += after.attention_cpu_fallbacks
                                                 - before.counters.attention_cpu_fallbacks;
    statistics.vulkan_shared_expert_swiglu_fusions += after.shared_expert_swiglu_fusions - before.counters.shared_expert_swiglu_fusions;
    statistics.vulkan_gated_delta_fusions += after.gated_delta_fusions - before.counters.gated_delta_fusions;
    statistics.vulkan_gated_delta_submissions += after.gated_delta_submissions - before.counters.gated_delta_submissions;
    statistics.vulkan_rms_norm_linear_fusions += after.rms_norm_linear_fusions - before.counters.rms_norm_linear_fusions;
    statistics.vulkan_kv_ring_appends += after.kv_ring_appends - before.counters.kv_ring_appends;
    statistics.vulkan_kv_ring_resizes += after.kv_ring_resizes - before.counters.kv_ring_resizes;
    statistics.vulkan_kv_ring_wrapped_views += after.kv_ring_wrapped_views - before.counters.kv_ring_wrapped_views;
    statistics.vulkan_kv_cache_promotions += after.kv_cache_promotions - before.counters.kv_cache_promotions;
    statistics.vulkan_kv_cache_promotion_bytes += after.kv_cache_promotion_bytes - before.counters.kv_cache_promotion_bytes;
    statistics.vulkan_bfloat16_cooperative_matrix_dispatches += after.bfloat16_cooperative_matrix_dispatches
                                                                - before.counters.bfloat16_cooperative_matrix_dispatches;
    statistics.vulkan_command_dispatches += after.command_dispatches - before.counters.command_dispatches;
    statistics.vulkan_command_pipeline_binds += after.command_pipeline_binds
                                                - before.counters.command_pipeline_binds;
    statistics.vulkan_command_redundant_pipeline_binds += after.command_redundant_pipeline_binds
                                                          - before.counters.command_redundant_pipeline_binds;
    statistics.vulkan_command_descriptor_bindings += after.command_descriptor_bindings
                                                     - before.counters.command_descriptor_bindings;
    statistics.vulkan_command_push_constant_updates += after.command_push_constant_updates
                                                       - before.counters.command_push_constant_updates;
    statistics.vulkan_command_resource_barrier_calls += after.command_resource_barrier_calls
                                                        - before.counters.command_resource_barrier_calls;
    statistics.vulkan_command_buffer_resource_barriers += after.command_buffer_resource_barriers
                                                          - before.counters.command_buffer_resource_barriers;
    statistics.vulkan_command_image_resource_barriers += after.command_image_resource_barriers
                                                         - before.counters.command_image_resource_barriers;
}

void record_expert_backend_delta(SessionStatistics& statistics, const ExpertBackendStatistics& before, const ExpertBackendStatistics& after)
{
    statistics.expert_gpu_cache_hits += after.hits - before.hits;
    statistics.expert_gpu_cache_misses += after.misses - before.misses;
    statistics.expert_gpu_cache_admissions += after.admissions - before.admissions;
    statistics.expert_gpu_cache_stores += after.stores - before.stores;
    statistics.expert_gpu_cache_evictions += after.evictions - before.evictions;
    statistics.expert_gpu_cache_dropped_admissions += after.dropped_admissions - before.dropped_admissions;
    statistics.expert_gpu_cache_bytes_uploaded += after.bytes_uploaded - before.bytes_uploaded;
    statistics.expert_gpu_cache_resident_size = after.resident_size;
    statistics.expert_gpu_cache_pending_size = after.pending_size;
    statistics.expert_gpu_executions += after.executions - before.executions;
    statistics.expert_gpu_execution_failures += after.execution_failures - before.execution_failures;
    statistics.expert_gpu_execution_time_microseconds += after.execution_time_microseconds - before.execution_time_microseconds;
    statistics.expert_gpu_arc_recent_size = after.arc_recent_size;
    statistics.expert_gpu_arc_frequent_size = after.arc_frequent_size;
    statistics.expert_gpu_arc_recent_target_size = after.arc_recent_target_size;
    statistics.expert_gpu_arc_recent_ghost_size = after.arc_recent_ghost_size;
    statistics.expert_gpu_arc_frequent_ghost_size = after.arc_frequent_ghost_size;
    statistics.expert_gpu_device_source_hits += after.device_source_hits - before.device_source_hits;
    statistics.expert_gpu_device_source_misses += after.device_source_misses - before.device_source_misses;
    statistics.expert_gpu_device_source_executions += after.device_source_executions - before.device_source_executions;
    statistics.expert_gpu_device_source_execution_failures += after.device_source_execution_failures - before.device_source_execution_failures;
    statistics.expert_gpu_route_aggregation_batches += after.route_aggregation_batches - before.route_aggregation_batches;
    statistics.expert_gpu_route_aggregation_routes += after.route_aggregation_routes - before.route_aggregation_routes;
    statistics.expert_gpu_route_aggregation_bytes_saved += after.route_aggregation_bytes_saved - before.route_aggregation_bytes_saved;
}

void record_expert_victim_cache_delta(SessionStatistics& statistics, const ExpertVictimCacheStatistics& before, const ExpertVictimCacheStatistics& after)
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
    statistics.expert_gpu_victim_cache_resident_size = after.resident_size;
    statistics.expert_gpu_victim_cache_pending_size = after.pending_size;
}

static uint64_t counter_delta(uint64_t current, uint64_t baseline) noexcept
{
    return current >= baseline ? current - baseline : 0;
}

static RuntimeMetricCounters runtime_metric_counters(
    const SessionStatistics& statistics,
    const SessionStatistics* baseline)
{
    const SessionStatistics empty;
    const SessionStatistics& start = baseline == nullptr ? empty : *baseline;
    RuntimeMetricCounters result;
    result.prefill_tokens = counter_delta(statistics.prefill_tokens, start.prefill_tokens);
    result.decode_tokens = counter_delta(statistics.decode_tokens, start.decode_tokens);
    result.expert_cache_hits = counter_delta(statistics.expert_cache_hits, start.expert_cache_hits);
    result.expert_cache_misses = counter_delta(statistics.expert_cache_misses, start.expert_cache_misses);
    result.expert_io_bytes = counter_delta(statistics.expert_cache_bytes_read, start.expert_cache_bytes_read);
    result.expert_compute_time_microseconds = counter_delta(
        statistics.expert_compute_time_microseconds,
        start.expert_compute_time_microseconds);
    result.gpu_submit_count = counter_delta(
        statistics.vulkan_compute_submissions,
        start.vulkan_compute_submissions);
    result.gpu_wait_time_microseconds = counter_delta(
        statistics.vulkan_submit_wait_time_microseconds,
        start.vulkan_submit_wait_time_microseconds);
    result.gpu_kernel_time_microseconds = counter_delta(
        statistics.expert_gpu_execution_time_microseconds,
        start.expert_gpu_execution_time_microseconds);
    result.gpu_kernel_time_available = counter_delta(
                                           statistics.expert_gpu_executions,
                                           start.expert_gpu_executions)
                                       != 0;
    result.vulkan_linear_dispatches = counter_delta(
        statistics.vulkan_linear_dispatches,
        start.vulkan_linear_dispatches);
    result.vulkan_attention_blocks = counter_delta(
        statistics.vulkan_attention_blocks,
        start.vulkan_attention_blocks);
    result.vulkan_batch_uploads = counter_delta(
        statistics.vulkan_batch_uploads,
        start.vulkan_batch_uploads);
    result.vulkan_batch_downloads = counter_delta(
        statistics.vulkan_batch_downloads,
        start.vulkan_batch_downloads);
    result.vulkan_attention_qkv_rope_fusions = counter_delta(
        statistics.vulkan_attention_qkv_rope_fusions,
        start.vulkan_attention_qkv_rope_fusions);
    result.vulkan_attention_device_rope_fusions = counter_delta(
        statistics.vulkan_attention_device_rope_fusions,
        start.vulkan_attention_device_rope_fusions);
    result.vulkan_attention_qkv_ring_fusions = counter_delta(
        statistics.vulkan_attention_qkv_ring_fusions,
        start.vulkan_attention_qkv_ring_fusions);
    result.vulkan_attention_qkv_rope_pipeline_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_pipeline_failures,
        start.vulkan_attention_qkv_rope_pipeline_failures);
    result.vulkan_attention_qkv_rope_shape_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_shape_failures,
        start.vulkan_attention_qkv_rope_shape_failures);
    result.vulkan_attention_qkv_rope_source_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_source_failures,
        start.vulkan_attention_qkv_rope_source_failures);
    result.vulkan_attention_qkv_rope_norm_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_norm_failures,
        start.vulkan_attention_qkv_rope_norm_failures);
    result.vulkan_attention_qkv_rope_ring_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_ring_failures,
        start.vulkan_attention_qkv_rope_ring_failures);
    result.vulkan_attention_qkv_rope_allocation_failures = counter_delta(
        statistics.vulkan_attention_qkv_rope_allocation_failures,
        start.vulkan_attention_qkv_rope_allocation_failures);
    result.vulkan_attention_precondition_failures = counter_delta(
        statistics.vulkan_attention_precondition_failures,
        start.vulkan_attention_precondition_failures);
    result.vulkan_attention_staging_failures = counter_delta(
        statistics.vulkan_attention_staging_failures,
        start.vulkan_attention_staging_failures);
    result.vulkan_attention_norm_failures = counter_delta(
        statistics.vulkan_attention_norm_failures,
        start.vulkan_attention_norm_failures);
    result.vulkan_attention_qkv_failures = counter_delta(
        statistics.vulkan_attention_qkv_failures,
        start.vulkan_attention_qkv_failures);
    result.vulkan_attention_cache_failures = counter_delta(
        statistics.vulkan_attention_cache_failures,
        start.vulkan_attention_cache_failures);
    result.vulkan_attention_sdpa_failures = counter_delta(
        statistics.vulkan_attention_sdpa_failures,
        start.vulkan_attention_sdpa_failures);
    result.vulkan_attention_projection_failures = counter_delta(
        statistics.vulkan_attention_projection_failures,
        start.vulkan_attention_projection_failures);
    result.vulkan_attention_output_failures = counter_delta(
        statistics.vulkan_attention_output_failures,
        start.vulkan_attention_output_failures);
    result.vulkan_attention_submit_failures = counter_delta(
        statistics.vulkan_attention_submit_failures,
        start.vulkan_attention_submit_failures);
    result.vulkan_attention_cache_materializations = counter_delta(
        statistics.vulkan_attention_cache_materializations,
        start.vulkan_attention_cache_materializations);
    result.vulkan_attention_cpu_fallbacks = counter_delta(
        statistics.vulkan_attention_cpu_fallbacks,
        start.vulkan_attention_cpu_fallbacks);
    result.vulkan_gated_delta_fusions = counter_delta(
        statistics.vulkan_gated_delta_fusions,
        start.vulkan_gated_delta_fusions);
    result.vulkan_gated_delta_submissions = counter_delta(
        statistics.vulkan_gated_delta_submissions,
        start.vulkan_gated_delta_submissions);
    result.expert_gpu_cache_hits = counter_delta(
        statistics.expert_gpu_cache_hits,
        start.expert_gpu_cache_hits);
    result.expert_gpu_cache_misses = counter_delta(
        statistics.expert_gpu_cache_misses,
        start.expert_gpu_cache_misses);
    result.expert_gpu_cache_admissions = counter_delta(
        statistics.expert_gpu_cache_admissions,
        start.expert_gpu_cache_admissions);
    result.expert_gpu_cache_stores = counter_delta(
        statistics.expert_gpu_cache_stores,
        start.expert_gpu_cache_stores);
    result.expert_gpu_cache_dropped_admissions = counter_delta(
        statistics.expert_gpu_cache_dropped_admissions,
        start.expert_gpu_cache_dropped_admissions);
    result.expert_gpu_cache_resident_size = statistics.expert_gpu_cache_resident_size;
    result.expert_gpu_cache_pending_size = statistics.expert_gpu_cache_pending_size;
    result.expert_gpu_executions = counter_delta(
        statistics.expert_gpu_executions,
        start.expert_gpu_executions);
    result.expert_gpu_execution_failures = counter_delta(
        statistics.expert_gpu_execution_failures,
        start.expert_gpu_execution_failures);
    result.expert_gpu_route_aggregation_batches = counter_delta(
        statistics.expert_gpu_route_aggregation_batches,
        start.expert_gpu_route_aggregation_batches);
    result.expert_gpu_route_aggregation_routes = counter_delta(
        statistics.expert_gpu_route_aggregation_routes,
        start.expert_gpu_route_aggregation_routes);
    result.expert_gpu_route_aggregation_bytes_saved = counter_delta(
        statistics.expert_gpu_route_aggregation_bytes_saved,
        start.expert_gpu_route_aggregation_bytes_saved);
    result.expert_cache_resident_size = statistics.expert_cache_resident_size;
    result.kv_cache_logical_size = statistics.kv_cache_logical_size;
    result.kv_cache_allocated_size = statistics.kv_cache_allocated_size;
    return result;
}

void Session::begin_generation(uint64_t input_tokens)
{
    generation_active = true;
    generation_input_tokens = input_tokens;
    generation_output_tokens = 0;
    generation_elapsed_microseconds = 0;
    generation_start_time = std::chrono::steady_clock::now();
    generation_first_token_time = {};
    generation_start_stats = stats;
}

void Session::finish_generation() noexcept
{
    if (!generation_active)
        return;
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - generation_start_time);
    generation_elapsed_microseconds = static_cast<uint64_t>(elapsed.count());
    generation_active = false;
}

SessionMetrics Session::metrics_unlocked() const
{
    SessionMetrics result;
    result.generation = runtime_metric_counters(stats, &generation_start_stats);
    result.cumulative = runtime_metric_counters(stats, nullptr);
    result.gpu_available = model->hybrid_mode() != HybridMode::CpuOnly;
    result.timing.active = generation_active;
    result.timing.input_tokens = generation_input_tokens;
    result.timing.output_tokens = generation_output_tokens;

    uint64_t elapsed_microseconds = generation_elapsed_microseconds;
    if (generation_active)
    {
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - generation_start_time);
        elapsed_microseconds = static_cast<uint64_t>(elapsed.count());
    }
    result.timing.elapsed_microseconds = elapsed_microseconds;

    uint64_t ttft_microseconds = 0;
    if (generation_output_tokens != 0)
    {
        const auto ttft = std::chrono::duration_cast<std::chrono::microseconds>(
            generation_first_token_time - generation_start_time);
        ttft_microseconds = static_cast<uint64_t>(ttft.count());
        result.timing.prompt_elapsed_microseconds = ttft_microseconds;
        result.timing.ttft_microseconds = ttft_microseconds;
        if (generation_input_tokens != 0 && ttft_microseconds != 0)
        {
            result.timing.prompt_tokens_per_second = static_cast<double>(generation_input_tokens) * 1000000.0
                                                     / static_cast<double>(ttft_microseconds);
        }
    }

    if (generation_output_tokens != 0 && elapsed_microseconds >= ttft_microseconds)
    {
        const uint64_t decode_elapsed_microseconds = elapsed_microseconds - ttft_microseconds;
        result.timing.generation_elapsed_microseconds = decode_elapsed_microseconds;
        if (generation_output_tokens > 1 && decode_elapsed_microseconds != 0)
        {
            const uint64_t decode_tokens = generation_output_tokens - 1;
            result.timing.tpot_microseconds = static_cast<double>(decode_elapsed_microseconds) / static_cast<double>(decode_tokens);
            result.timing.generation_tokens_per_second = static_cast<double>(decode_tokens) * 1000000.0 / static_cast<double>(decode_elapsed_microseconds);
            result.timing.decode_tokens_per_second = result.timing.generation_tokens_per_second;
        }
    }
    return result;
}

SessionMetrics Session::metrics() const
{
    const std::lock_guard<std::mutex> lock(mutex);
    return metrics_unlocked();
}

} // namespace moe
} // namespace ncnn
