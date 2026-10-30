#include "metrics.h"

#include "backends/ncnn/vulkan.h"
#include "executor.h"
#include "expertbackend.h"
#include "ncnn/moe/session.h"
#include "storage/expertcache_victim.h"
#include "storage/expertcache.h"
#include "graph/layerplan.h"
#include "graph/compiledmodel.h"

#include <chrono>

namespace ncnn {
namespace moe {

static void add_cache_delta(SessionStatistics& stats,
                            const ExpertCacheStatistics& before,
                            const ExpertCacheStatistics& after)
{
    stats.expert_cache_hits += after.hits - before.hits;
    stats.expert_cache_misses += after.misses - before.misses;
    stats.expert_cache_evictions += after.evictions - before.evictions;
    stats.expert_cache_bytes_read += after.bytes_read - before.bytes_read;
    stats.expert_cache_queued_reads += after.queued_reads - before.queued_reads;
    stats.expert_cache_speculative_reads += after.speculative_reads - before.speculative_reads;
    stats.expert_cache_cancelled_speculative_reads += after.cancelled_speculative_reads - before.cancelled_speculative_reads;
    stats.expert_cache_dropped_speculative_admissions += after.dropped_speculative_admissions - before.dropped_speculative_admissions;
    stats.expert_cache_unused_speculative_reads += after.unused_speculative_reads - before.unused_speculative_reads;
    stats.expert_cache_short_term_reloads += after.short_term_reloads - before.short_term_reloads;
    stats.expert_cache_arc_recent_size = after.arc_recent_size;
    stats.expert_cache_arc_frequent_size = after.arc_frequent_size;
    stats.expert_cache_arc_recent_target_size = after.arc_recent_target_size;
    stats.expert_cache_arc_recent_ghost_size = after.arc_recent_ghost_size;
    stats.expert_cache_arc_frequent_ghost_size = after.arc_frequent_ghost_size;
    stats.expert_cache_arc_recent_ghost_hits += after.arc_recent_ghost_hits - before.arc_recent_ghost_hits;
    stats.expert_cache_arc_frequent_ghost_hits += after.arc_frequent_ghost_hits - before.arc_frequent_ghost_hits;
    stats.expert_cache_mapped_ranges += after.mapped_ranges - before.mapped_ranges;
    stats.expert_cache_mapped_bytes += after.mapped_bytes - before.mapped_bytes;
    stats.expert_cache_direct_read_ranges += after.direct_read_ranges - before.direct_read_ranges;
    stats.expert_cache_direct_read_bytes += after.direct_read_bytes - before.direct_read_bytes;
    stats.expert_cache_direct_read_fallbacks += after.direct_read_fallbacks - before.direct_read_fallbacks;
    stats.expert_cache_buffered_read_ranges += after.buffered_read_ranges - before.buffered_read_ranges;
    stats.expert_cache_buffered_read_bytes += after.buffered_read_bytes - before.buffered_read_bytes;
    stats.expert_cache_coalesced_read_batches += after.coalesced_read_batches - before.coalesced_read_batches;
    stats.expert_cache_coalesced_experts += after.coalesced_experts - before.coalesced_experts;
    stats.expert_cache_coalesced_read_ranges_saved += after.coalesced_read_ranges_saved - before.coalesced_read_ranges_saved;
    stats.expert_cache_read_policy = after.adaptive_read_policy;
    stats.expert_cache_num_io_threads = after.num_io_threads;
    stats.expert_cache_resident_size = after.resident_size;
    stats.expert_gpu_victim_cache_hits += after.victim.hits - before.victim.hits;
    stats.expert_gpu_victim_cache_misses += after.victim.misses - before.victim.misses;
    stats.expert_gpu_victim_cache_admissions += after.victim.admissions - before.victim.admissions;
    stats.expert_gpu_victim_cache_filtered_admissions += after.victim.filtered_admissions - before.victim.filtered_admissions;
    stats.expert_gpu_victim_cache_reused_admissions += after.victim.reused_admissions - before.victim.reused_admissions;
    stats.expert_gpu_victim_cache_probe_admissions += after.victim.probe_admissions - before.victim.probe_admissions;
    stats.expert_gpu_victim_cache_stores += after.victim.stores - before.victim.stores;
    stats.expert_gpu_victim_cache_evictions += after.victim.evictions - before.victim.evictions;
    stats.expert_gpu_victim_cache_dropped_admissions += after.victim.dropped_admissions - before.victim.dropped_admissions;
    stats.expert_gpu_victim_cache_restore_failures += after.victim.restore_failures - before.victim.restore_failures;
    stats.expert_gpu_victim_cache_bytes_uploaded += after.victim.bytes_uploaded - before.victim.bytes_uploaded;
    stats.expert_gpu_victim_cache_bytes_downloaded += after.victim.bytes_downloaded - before.victim.bytes_downloaded;
    stats.expert_gpu_victim_cache_mapped_stores += after.victim.mapped_stores - before.victim.mapped_stores;
    stats.expert_gpu_victim_cache_mapped_restores += after.victim.mapped_restores - before.victim.mapped_restores;
    stats.expert_gpu_victim_cache_resident_size = after.victim.resident_size;
    stats.expert_gpu_victim_cache_pending_size = after.victim.pending_size;
}

void record_vulkan_delta(SessionStatistics& stats,
                         const VulkanStatistics& before,
                         const VulkanStatistics& after)
{
    stats.vulkan_linear_dispatches += after.dispatches - before.dispatches;
    stats.vulkan_int4_dispatches += after.int4_dispatches - before.int4_dispatches;
    stats.vulkan_attention_blocks += after.attention_blocks - before.attention_blocks;
    stats.vulkan_compute_submissions += after.compute_submissions - before.compute_submissions;
    stats.vulkan_batch_uploads += after.batch_uploads - before.batch_uploads;
    stats.vulkan_batch_downloads += after.batch_downloads - before.batch_downloads;
    stats.vulkan_attention_qkv_rope_fusions += after.attention_qkv_rope_fusions - before.attention_qkv_rope_fusions;
    stats.vulkan_attention_device_rope_fusions += after.attention_device_rope_fusions - before.attention_device_rope_fusions;
    stats.vulkan_attention_qkv_ring_fusions += after.attention_qkv_ring_fusions - before.attention_qkv_ring_fusions;
    stats.vulkan_attention_decode_sdpa_fusions += after.attention_decode_sdpa_fusions - before.attention_decode_sdpa_fusions;
    stats.vulkan_attention_cache_materializations += after.attention_cache_materializations
                                                     - before.attention_cache_materializations;
    stats.vulkan_attention_cpu_fallbacks += after.attention_cpu_fallbacks
                                            - before.attention_cpu_fallbacks;
    stats.vulkan_shared_expert_swiglu_fusions += after.shared_expert_swiglu_fusions - before.shared_expert_swiglu_fusions;
    stats.vulkan_gated_delta_fusions += after.gated_delta_fusions - before.gated_delta_fusions;
    stats.vulkan_gated_delta_submissions += after.gated_delta_submissions - before.gated_delta_submissions;
    stats.vulkan_rms_norm_linear_fusions += after.rms_norm_linear_fusions - before.rms_norm_linear_fusions;
    stats.vulkan_kv_ring_appends += after.kv_ring_appends - before.kv_ring_appends;
    stats.vulkan_kv_ring_resizes += after.kv_ring_resizes - before.kv_ring_resizes;
    stats.vulkan_kv_ring_wrapped_views += after.kv_ring_wrapped_views - before.kv_ring_wrapped_views;
    stats.vulkan_bfloat16_cooperative_matrix_dispatches += after.bfloat16_cooperative_matrix_dispatches
                                                           - before.bfloat16_cooperative_matrix_dispatches;
}

static void add_backend_delta(SessionStatistics& stats,
                              const ExpertBackendStatistics& before,
                              const ExpertBackendStatistics& after)
{
    stats.expert_gpu_cache_hits += after.hits - before.hits;
    stats.expert_gpu_cache_misses += after.misses - before.misses;
    stats.expert_gpu_cache_admissions += after.admissions - before.admissions;
    stats.expert_gpu_cache_stores += after.stores - before.stores;
    stats.expert_gpu_cache_evictions += after.evictions - before.evictions;
    stats.expert_gpu_cache_dropped_admissions += after.dropped_admissions - before.dropped_admissions;
    stats.expert_gpu_cache_bytes_uploaded += after.bytes_uploaded - before.bytes_uploaded;
    stats.expert_gpu_cache_resident_size = after.resident_size;
    stats.expert_gpu_cache_pending_size = after.pending_size;
    stats.expert_gpu_executions += after.executions - before.executions;
    stats.expert_gpu_execution_failures += after.execution_failures - before.execution_failures;
    stats.expert_gpu_arc_recent_size = after.arc_recent_size;
    stats.expert_gpu_arc_frequent_size = after.arc_frequent_size;
    stats.expert_gpu_arc_recent_target_size = after.arc_recent_target_size;
    stats.expert_gpu_arc_recent_ghost_size = after.arc_recent_ghost_size;
    stats.expert_gpu_arc_frequent_ghost_size = after.arc_frequent_ghost_size;
    stats.expert_gpu_device_source_hits += after.device_source_hits - before.device_source_hits;
    stats.expert_gpu_device_source_misses += after.device_source_misses - before.device_source_misses;
    stats.expert_gpu_device_source_executions += after.device_source_executions - before.device_source_executions;
    stats.expert_gpu_device_source_execution_failures += after.device_source_execution_failures - before.device_source_execution_failures;
    stats.expert_gpu_route_aggregation_batches += after.route_aggregation_batches - before.route_aggregation_batches;
    stats.expert_gpu_route_aggregation_routes += after.route_aggregation_routes - before.route_aggregation_routes;
    stats.expert_gpu_route_aggregation_bytes_saved += after.route_aggregation_bytes_saved - before.route_aggregation_bytes_saved;
}

void record_resource_delta(const CompiledModel& model,
                           SessionStatistics& stats,
                           const ExpertCacheStatistics& cache_before,
                           const ExpertBackendStatistics& backend_before)
{
    if (model.expert_cache)
    {
        const ExpertCacheStatistics cache_after = model.expert_cache->statistics();
        add_cache_delta(stats, cache_before, cache_after);
    }
    if (model.expert_backend)
    {
        add_backend_delta(stats, backend_before, model.expert_backend->statistics());
    }
}

void record_resource_delta(const CompiledModel& model,
                           std::span<const DecodeBatchEntry> entries,
                           const ExpertCacheStatistics& cache_before,
                           const ExpertBackendStatistics& backend_before)
{
    // Each participating Session retains the logical batch resource delta.
    if (model.expert_cache)
    {
        const ExpertCacheStatistics cache_after = model.expert_cache->statistics();
        for (const DecodeBatchEntry& entry : entries)
        {
            SessionStatistics& stats = *entry.statistics;
            add_cache_delta(stats, cache_before, cache_after);
        }
    }
    if (model.expert_backend)
    {
        const ExpertBackendStatistics backend_after = model.expert_backend->statistics();
        for (const DecodeBatchEntry& entry : entries)
        {
            add_backend_delta(*entry.statistics, backend_before, backend_after);
        }
    }
}

static uint64_t counter_delta(uint64_t current, uint64_t baseline) noexcept
{
    return current >= baseline ? current - baseline : 0;
}

RuntimeMetricCounters metric_counters(const SessionStatistics& stats,
                                      const RuntimeMetricCounters* baseline)
{
    const RuntimeMetricCounters empty;
    const RuntimeMetricCounters& start = baseline == nullptr ? empty : *baseline;
    RuntimeMetricCounters result;
    result.prefill_tokens = counter_delta(stats.prefill_tokens, start.prefill_tokens);
    result.decode_tokens = counter_delta(stats.decode_tokens, start.decode_tokens);
    result.expert_assignments = counter_delta(stats.expert_assignments, start.expert_assignments);
    result.expert_batches = counter_delta(stats.expert_batches, start.expert_batches);
    result.expert_cpu_batches = counter_delta(stats.expert_cpu_batches, start.expert_cpu_batches);
    result.expert_cpu_routes = counter_delta(stats.expert_cpu_routes, start.expert_cpu_routes);
    result.expert_gpu_routes = counter_delta(stats.expert_gpu_routes, start.expert_gpu_routes);
    result.expert_cpu_fallbacks = counter_delta(stats.expert_cpu_fallbacks, start.expert_cpu_fallbacks);
    result.vulkan_qwen_layers = counter_delta(stats.vulkan_qwen_layers, start.vulkan_qwen_layers);
    result.vulkan_gated_residual_blocks = counter_delta(stats.vulkan_gated_residual_blocks, start.vulkan_gated_residual_blocks);
    result.vulkan_ple_blocks = counter_delta(stats.vulkan_ple_blocks, start.vulkan_ple_blocks);
    result.expert_cache_hits = counter_delta(stats.expert_cache_hits, start.expert_cache_hits);
    result.expert_cache_misses = counter_delta(stats.expert_cache_misses, start.expert_cache_misses);
    result.expert_io_bytes = counter_delta(stats.expert_cache_bytes_read, start.expert_io_bytes);
    result.gpu_submit_count = counter_delta(stats.vulkan_compute_submissions,
                                            start.gpu_submit_count);
    result.vulkan_linear_dispatches = counter_delta(stats.vulkan_linear_dispatches,
                                                    start.vulkan_linear_dispatches);
    result.vulkan_int4_dispatches = counter_delta(stats.vulkan_int4_dispatches,
                                                  start.vulkan_int4_dispatches);
    result.vulkan_attention_blocks = counter_delta(stats.vulkan_attention_blocks,
                                                   start.vulkan_attention_blocks);
    result.vulkan_batch_uploads = counter_delta(stats.vulkan_batch_uploads,
                                                start.vulkan_batch_uploads);
    result.vulkan_batch_downloads = counter_delta(stats.vulkan_batch_downloads,
                                                  start.vulkan_batch_downloads);
    result.vulkan_attention_qkv_rope_fusions = counter_delta(stats.vulkan_attention_qkv_rope_fusions,
                                                             start.vulkan_attention_qkv_rope_fusions);
    result.vulkan_attention_device_rope_fusions = counter_delta(stats.vulkan_attention_device_rope_fusions,
                                                                start.vulkan_attention_device_rope_fusions);
    result.vulkan_attention_qkv_ring_fusions = counter_delta(stats.vulkan_attention_qkv_ring_fusions,
                                                             start.vulkan_attention_qkv_ring_fusions);
    result.vulkan_attention_cache_materializations = counter_delta(stats.vulkan_attention_cache_materializations,
                                                                   start.vulkan_attention_cache_materializations);
    result.vulkan_attention_cpu_fallbacks = counter_delta(stats.vulkan_attention_cpu_fallbacks,
                                                          start.vulkan_attention_cpu_fallbacks);
    result.vulkan_gated_delta_fusions = counter_delta(stats.vulkan_gated_delta_fusions,
                                                      start.vulkan_gated_delta_fusions);
    result.vulkan_gated_delta_submissions = counter_delta(stats.vulkan_gated_delta_submissions,
                                                          start.vulkan_gated_delta_submissions);
    result.expert_gpu_cache_hits = counter_delta(stats.expert_gpu_cache_hits,
                                                 start.expert_gpu_cache_hits);
    result.speculative_proposals = counter_delta(stats.speculative_proposals, start.speculative_proposals);
    result.speculative_draft_tokens = counter_delta(stats.speculative_draft_tokens, start.speculative_draft_tokens);
    result.speculative_accepted_tokens = counter_delta(stats.speculative_accepted_tokens, start.speculative_accepted_tokens);
    result.expert_gpu_cache_misses = counter_delta(stats.expert_gpu_cache_misses,
                                                   start.expert_gpu_cache_misses);
    result.expert_gpu_cache_admissions = counter_delta(stats.expert_gpu_cache_admissions,
                                                       start.expert_gpu_cache_admissions);
    result.expert_gpu_cache_stores = counter_delta(stats.expert_gpu_cache_stores,
                                                   start.expert_gpu_cache_stores);
    result.expert_gpu_cache_evictions = counter_delta(stats.expert_gpu_cache_evictions,
                                                      start.expert_gpu_cache_evictions);
    result.expert_gpu_cache_dropped_admissions = counter_delta(stats.expert_gpu_cache_dropped_admissions,
                                                               start.expert_gpu_cache_dropped_admissions);
    result.expert_gpu_cache_resident_size = stats.expert_gpu_cache_resident_size;
    result.expert_gpu_cache_pending_size = stats.expert_gpu_cache_pending_size;
    result.expert_gpu_executions = counter_delta(stats.expert_gpu_executions,
                                                 start.expert_gpu_executions);
    result.expert_gpu_execution_failures = counter_delta(stats.expert_gpu_execution_failures,
                                                         start.expert_gpu_execution_failures);
    result.expert_gpu_route_aggregation_batches = counter_delta(stats.expert_gpu_route_aggregation_batches,
                                                                start.expert_gpu_route_aggregation_batches);
    result.expert_gpu_route_aggregation_routes = counter_delta(stats.expert_gpu_route_aggregation_routes,
                                                               start.expert_gpu_route_aggregation_routes);
    result.expert_gpu_route_aggregation_bytes_saved = counter_delta(stats.expert_gpu_route_aggregation_bytes_saved,
                                                                    start.expert_gpu_route_aggregation_bytes_saved);
    result.expert_cache_resident_size = stats.expert_cache_resident_size;
    result.kv_cache_logical_size = stats.kv_cache_logical_size;
    result.kv_cache_allocated_size = stats.kv_cache_allocated_size;
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
    metrics_base = metric_counters(stats, nullptr);
    metrics_valid = true;
}

void Session::finish_generation() noexcept
{
    if (!generation_active)
        return;
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - generation_start_time);
    generation_elapsed_microseconds = static_cast<uint64_t>(elapsed.count());
    generation_active = false;
}

SessionMetrics Session::metrics() const
{
    const std::lock_guard<std::mutex> lock(mutex);
    SessionMetrics result;
    result.cumulative = metric_counters(stats, nullptr);
    const RuntimeMetricCounters& baseline = metrics_valid ? metrics_base : result.cumulative;
    result.generation = metric_counters(stats, &baseline);
    result.gpu_available = model->hybrid_mode() != HybridMode::CpuOnly;
    result.timing.active = generation_active;
    result.timing.input_tokens = generation_input_tokens;
    result.timing.output_tokens = generation_output_tokens;

    uint64_t elapsed_microseconds = generation_elapsed_microseconds;
    if (generation_active)
    {
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - generation_start_time);
        elapsed_microseconds = static_cast<uint64_t>(elapsed.count());
    }
    result.timing.elapsed_microseconds = elapsed_microseconds;

    uint64_t ttft_microseconds = 0;
    if (generation_output_tokens != 0)
    {
        const auto ttft = std::chrono::duration_cast<std::chrono::microseconds>(generation_first_token_time - generation_start_time);
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

} // namespace moe
} // namespace ncnn
