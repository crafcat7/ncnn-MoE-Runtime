#include "executor.h"

#include "kernels/attention.h"
#include "kernels/activationbuffer.h"
#include "kernels/gateddeltanet.h"
#include "kernels/gatedresidual.h"
#include "kernels/hyperconnection.h"
#include "kernels/latentattention.h"
#include "kernels/ops.h"
#include "kernels/ple.h"
#include "sessionstate.h"
#include "cpu.h"
#include "expertbackend.h"
#include "expert.h"
#include "graphresources.h"
#include "storage/expertcache.h"
#include "backends/ncnn/modelpipeline.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/latentlayer_vulkan.h"
#include "backends/ncnn/moecombine_vulkan.h"
#include "backends/ncnn/residentffn_vulkan.h"
#include "backends/ncnn/layerhead_vulkan.h"
#include "backends/ncnn/gatedresiduallayer_vulkan.h"
#include "backends/ncnn/gpusegment_vulkan.h"

#include "graph/router.h"
#include "graph/compiledmodel.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <future>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

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
    stats.vulkan_attention_cpu_fallbacks += after.attention_cpu_fallbacks
                                            - before.attention_cpu_fallbacks;
    stats.vulkan_kv_ring_appends += after.kv_ring_appends - before.kv_ring_appends;
    stats.vulkan_kv_ring_resizes += after.kv_ring_resizes - before.kv_ring_resizes;
    stats.vulkan_kv_ring_wrapped_views += after.kv_ring_wrapped_views - before.kv_ring_wrapped_views;
    stats.vulkan_gated_delta_submissions += after.gated_delta_submissions - before.gated_delta_submissions;
}

struct ScopedResidentContinuation
{
    std::shared_ptr<GpuSegmentContinuation_vulkan>& pending;
    std::shared_ptr<ResidentFfnRegion_vulkan>& region;

    ~ScopedResidentContinuation()
    {
        // Region/worker owners may outlive this local reference. Invalidate
        // submitted, unpublished state now, before another Session operation.
        if (pending) pending->abort();
        region.reset();
    }
};

static Result<void> open_region(const CompiledModel& model, LayerState& state,
                                ExecutionBackend backend)
{
    if (backend != ExecutionBackend::Vulkan || model.opt.hybrid_mode != HybridMode::HybridExperts
        || (state.normalized_device && !state.gpu_segment) || state.region
        || !model.expert_backend || !model.expert_backend->supports_resident_region()
        || state.active_experts().empty())
        return {};
    state.region = state.gpu_segment
                       ? ResidentFfnRegion_vulkan::create(state.normalized_device, state.active_experts().size(), model.descriptor.hidden_size)
                       : ResidentFfnRegion_vulkan::create(state.normalized, state.active_experts().size(), model.descriptor.hidden_size);
    if (!state.region)
        return Error{ErrorCode::InternalError, "cannot create resident Vulkan FFN region"};
    return {};
}

struct RouterPredictionOutcome
{
    std::vector<uint32_t> predicted_expert_ids;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
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

struct ScopedLatentLayerWorkspace
{
    explicit ScopedLatentLayerWorkspace(LatentLayerWorkspace_vulkan& _workspace,
                                        std::vector<MoeCombineInput_vulkan>& _combine_inputs)
        : workspace(_workspace),
          combine_inputs(_combine_inputs)
    {
        combine_inputs.clear();
        workspace.reset();
    }
    ~ScopedLatentLayerWorkspace()
    {
        combine_inputs.clear();
        workspace.reset();
        // Bound inactive publication storage after a large prefill. Normal
        // decode/batch capacities remain reusable for the next execution.
        static constexpr uint64_t maximum_retained_host_size = 64 * 1024 * 1024;
        if (workspace.allocated_bytes() > maximum_retained_host_size)
            workspace.release();
    }
    LatentLayerWorkspace_vulkan& workspace;
    std::vector<MoeCombineInput_vulkan>& combine_inputs;
};

struct ScopedGatedResidualWorkspace
{
    GatedResidualWorkspace_vulkan& workspace;
    explicit ScopedGatedResidualWorkspace(GatedResidualWorkspace_vulkan& value) : workspace(value)
    { workspace.reset(); }
    ~ScopedGatedResidualWorkspace()
    { workspace.reset(); }
};

struct ScopedGpuSegmentWorkspace
{
    GpuSegmentWorkspace_vulkan& workspace;
    explicit ScopedGpuSegmentWorkspace(GpuSegmentWorkspace_vulkan& value) : workspace(value)
    { workspace.reset(); }
    ~ScopedGpuSegmentWorkspace()
    { workspace.reset(); }
};

static void resolve_routes(const CompiledModel& model,
                           const CompiledLayerPlan& layer,
                           const ExpertDispatchPlan& plan,
                           SessionState& state,
                           SessionStatistics& statistics,
                           bool resolve_unused_predictions);

[[nodiscard]] static Result<void> finish_predict(const CompiledModel& model,
                                                 uint32_t layer_id,
                                                 PendingRouterPrediction& pending,
                                                 SessionState& state,
                                                 SessionStatistics& statistics,
                                                 size_t token_count);

[[nodiscard]] static Result<void> prefetch_routes(const CompiledModel& model,
                                                  const CompiledLayerPlan& layer,
                                                  const ActivationBuffer& router_input,
                                                  SessionState& state,
                                                  SessionStatistics& statistics,
                                                  PendingRouterPrediction& pending);

static void capture_hidden(const ActivationBuffer& hidden,
                           uint32_t hidden_size,
                           uint32_t multiplier,
                           size_t target_index,
                           ActivationBuffer& destination)
{
    const float inverse_multiplier = 1.0f / static_cast<float>(multiplier);
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
        for (uint32_t column = 0; column < hidden_size; ++column)
            output[column] *= inverse_multiplier;
    }
}

static bool attn_state_unknown(const SessionState& state) noexcept
{
    return std::any_of(state.layers.begin(),
                       state.layers.end(),
                       [](const LayerCache& cache) {
                           return cache.vulkan_attention_state_unknown;
                       });
}

static void reset_state(const CompiledModel& model,
                        SessionState& state)
{
    if (state.layers.size() != model.graph.layer_plans.size())
        state.layers.resize(model.graph.layer_plans.size());
    state.execution_state.reset();
}

static bool shared_done(const LayerState& state) noexcept
{
    return state.shared_expert_output.rows() != 0 || static_cast<bool>(state.shared_expert_device_output);
}

static Result<void> fwd_shared(const CompiledModel& model,
                               const MoeBlockPlan& moe,
                               size_t rows,
                               LayerState& state,
                               bool retain_device_output,
                               ExecutionBackend backend)
{
    if (shared_done(state)) return {};
    if (state.normalized.rows() != rows
        || state.normalized.columns() != model.descriptor.hidden_size)
    {
        return Error{ErrorCode::InternalError, "Shared Expert input has an invalid shape"};
    }
    try
    {
        forward_shared_expert(model, moe, state.normalized, state.shared_expert_output,
                              state.shared_expert_workspace, state.normalized_device,
                              retain_device_output ? &state.shared_expert_device_output : nullptr,
                              state.require_device_execution,
                              backend == ExecutionBackend::Vulkan && model.opt.hybrid_mode == HybridMode::HybridExperts);
    }
    catch (const std::exception& error)
    {
        return Error{ErrorCode::InternalError, error.what()};
    }
    return {};
}

[[nodiscard]] static Result<void> fwd_shared_group(const CompiledModel& model,
                                                   const MoeBlockPlan& moe,
                                                   size_t rows,
                                                   LayerState& layer,
                                                   bool retain_device_output,
                                                   ExecutionBackend backend)
{
    if (layer.gpu_segment)
        return {};
    if (!moe.has_shared_expert)
        return Error{ErrorCode::InternalError, "Shared Expert graph node has no shared Expert plan"};
    return fwd_shared(model, moe, rows, layer, retain_device_output, backend);
}

static void router_linear(const CompiledModel& model,
                          const MoeBlockPlan& moe,
                          const ActivationBuffer& normalized,
                          ActivationBuffer& logits)
{
    forward_linear(model.weights.at(moe.router_weight),
                   normalized,
                   logits,
                   model.opt.optimization_flags,
                   model.operators.find_weight(moe.router_weight));
    if (moe.router_bias != invalid_tensor_handle)
        add_bias_inplace(logits, model.weights.at(moe.router_bias));
}

[[nodiscard]] static Result<void> router_fwd(const CompiledModel& model,
                                             const CompiledLayerPlan& layer,
                                             const MoeBlockPlan& moe,
                                             const ActivationBuffer& input,
                                             ActivationBuffer& normalized,
                                             ActivationBuffer& logits,
                                             HyperConnectionMix& mix,
                                             HyperConnectionScratch& hyper)
{
    if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
    {
        auto mixed = hc_pre(input,
                            model.weights.at(layer.hyper_connection.ffn_function),
                            model.weights.at(layer.hyper_connection.ffn_scale),
                            model.weights.at(layer.hyper_connection.ffn_base),
                            model.descriptor.hyper_connection_multiplier,
                            model.descriptor.hyper_connection_iterations,
                            model.descriptor.norm_epsilon,
                            model.descriptor.hyper_connection_epsilon,
                            mix,
                            hyper,
                            model.opt.optimization_flags);
        if (!mixed)
            return mixed.error();
        forward_rms_norm(mix.reduced,
                         model.weights.at(moe.pre_ffn_norm_weight),
                         model.descriptor.norm_epsilon,
                         normalized,
                         model.descriptor.norm_weight_offset);
    }
    else if (moe.pre_ffn_norm_bias != invalid_tensor_handle)
    {
        forward_layer_norm(input,
                           model.weights.at(moe.pre_ffn_norm_weight),
                           model.weights.at(moe.pre_ffn_norm_bias),
                           model.descriptor.norm_epsilon,
                           normalized);
    }
    else
    {
        forward_rms_norm(input,
                         model.weights.at(moe.pre_ffn_norm_weight),
                         model.descriptor.norm_epsilon,
                         normalized,
                         model.descriptor.norm_weight_offset);
    }
    router_linear(model, moe, normalized, logits);
    return {};
}

struct SharedOverlap
{
    const CompiledModel* model = nullptr;
    const MoeBlockPlan* moe = nullptr;
    LayerState* layer = nullptr;
    bool retain_device_output = false;
    ExecutionBackend backend = ExecutionBackend::Cpu;
    Result<void> result;
};

static void run_shared_overlap(void* raw)
{
    SharedOverlap& work = *static_cast<SharedOverlap*>(raw);
    work.result = fwd_shared(*work.model,
                             *work.moe,
                             work.layer->normalized.rows(),
                             *work.layer,
                             work.retain_device_output,
                             work.backend);
}

static Result<void> fwd_dense(const CompiledModel& model,
                              const CompiledLayerPlan& layer,
                              ActivationBuffer& hidden,
                              LayerState& scratch,
                              ExecutionBackend backend);

[[nodiscard]] static Result<void> fwd_dense_layer(const CompiledModel& model,
                                                  const CompiledLayerPlan& layer,
                                                  ActivationBuffer& hidden,
                                                  LayerState& layer_state,
                                                  GpuSegmentWorkspace_vulkan& gpu,
                                                  LatentLayerWorkspace_vulkan& latent,
                                                  ExecutionBackend backend)
{
    if (gpu.active())
    {
        std::string error;
        if (!GpuSegment_vulkan::materialize_hidden(gpu, hidden, error))
            return Error{ErrorCode::InternalError, "dense FFN GPU boundary failed: " + error};
    }
    if (latent.active() && !LatentLayer_vulkan::materialize_hidden(latent, hidden))
        return Error{ErrorCode::InternalError, "cannot materialize dense FFN input"};
    return fwd_dense(model, layer, hidden, layer_state, backend);
}

static Result<void> fwd_dense(const CompiledModel& model,
                              const CompiledLayerPlan& layer,
                              ActivationBuffer& hidden,
                              LayerState& scratch,
                              ExecutionBackend backend)
{
    if (layer.dense_pre_ffn_norm_bias != invalid_tensor_handle)
    {
        forward_layer_norm(hidden, model.weights.at(layer.dense_pre_ffn_norm_weight),
                           model.weights.at(layer.dense_pre_ffn_norm_bias), model.descriptor.norm_epsilon, scratch.normalized);
    }
    else
    {
        forward_rms_norm(hidden, model.weights.at(layer.dense_pre_ffn_norm_weight),
                         model.descriptor.norm_epsilon, scratch.normalized, model.descriptor.norm_weight_offset);
    }
    try
    {
        forward_dense_ffn(model, layer.dense_ffn, scratch.normalized, scratch.shared_expert_output,
                          scratch.shared_expert_workspace, backend);
    }
    catch (const std::exception& error)
    {
        return Error{ErrorCode::InternalError, error.what()};
    }
    add_batch_inplace(hidden, scratch.shared_expert_output);
    scratch.reset();
    return {};
}

static ExpertDispatchOptions route_opt(const CompiledModel& model,
                                       const MoeBlockPlan& moe,
                                       std::span<const int32_t> input_ids,
                                       std::vector<uint32_t>& explicit_ids)
{
    ExpertDispatchOptions options;
    options.expert_count = static_cast<uint32_t>(moe.experts.size());
    options.top_k = moe.top_k;
    options.group_count = moe.router_group_count;
    options.top_k_groups = moe.router_top_k_groups;
    options.score_function = moe.score_function;
    options.normalization = moe.normalization;
    options.routed_scaling_factor = moe.routed_scaling_factor;
    options.jitter = moe.jitter;
    if (moe.router_selection_bias != invalid_tensor_handle)
        options.selection_bias = model.weights.at(moe.router_selection_bias).float32_values();
    explicit_ids.clear();
    if (moe.token_experts != invalid_tensor_handle)
    {
        const auto table = model.weights.at(moe.token_experts).int64_values();
        explicit_ids.resize(input_ids.size() * moe.top_k);
        for (size_t token = 0; token < input_ids.size(); ++token)
            for (uint32_t rank = 0; rank < moe.top_k; ++rank)
                explicit_ids[token * moe.top_k + rank] = static_cast<uint32_t>(table[static_cast<size_t>(input_ids[token]) * moe.top_k + rank]);
        options.explicit_expert_ids = explicit_ids;
    }
    return options;
}

[[nodiscard]] static Result<void> combine_experts(const CompiledModel& model,
                                                  const MoeBlockPlan& moe,
                                                  LayerState& layer,
                                                  ExpertScratch& scratch,
                                                  ActivationBuffer& output,
                                                  size_t rows)
{
    if (moe.has_shared_expert && layer.shared_expert_device_output
        && !MoeCombine_vulkan::materialize(*layer.shared_expert_device_output, layer.shared_expert_output))
    {
        return Error{ErrorCode::InternalError, "cannot materialize resident Shared Expert output"};
    }
    for (ExpertState& active : layer.active_experts())
    {
        if (active.device_output && !MoeCombine_vulkan::materialize(*active.device_output, active.output))
            return Error{ErrorCode::InternalError, "cannot materialize resident Expert output"};
    }

    const bool has_aggregation = init_moe_output(scratch, rows, model.descriptor.hidden_size, output);
    for (size_t index = 0; index < layer.active_experts().size(); ++index)
    {
        const ExpertState& active = layer.active_experts()[index];
        if (has_aggregation && index < scratch.backend_aggregated.size()
            && scratch.backend_aggregated[index] != 0)
        {
            continue;
        }
        for (size_t route_index = 0; route_index < active.batch.routes.size(); ++route_index)
        {
            const ExpertRoute& route = active.batch.routes[route_index];
            if (route.token_index >= rows)
                return Error{ErrorCode::InternalError, "Expert route references an invalid row"};
            float* destination = output.row(route.token_index);
            const float* source = active.output.row(route_index);
            for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
                destination[column] += route.weight * source[column];
        }
    }
    if (moe.has_shared_expert)
        add_batch_inplace(output, layer.shared_expert_output);
    return {};
}

[[nodiscard]] static bool defer_experts(LayerState& layer,
                                        ExpertScratch& scratch,
                                        size_t rows,
                                        uint32_t hidden_size,
                                        LatentLayerWorkspace_vulkan& workspace)
{
    const ActivationBuffer* shared_host = nullptr;
    if (layer.shared_expert_output.rows() == rows
        && layer.shared_expert_output.columns() == hidden_size)
    {
        shared_host = &layer.shared_expert_output;
    }
    std::vector<MoeCombineInput_vulkan>& inputs = scratch.combine_inputs;
    inputs.clear();
    inputs.reserve(layer.active_experts().size());
    for (const ExpertState& active : layer.active_experts())
        inputs.push_back({&active.output, active.device_output, active.batch.routes});
    const bool combined = LatentLayer_vulkan::defer_combine(inputs, rows, hidden_size,
                                                            shared_host,
                                                            layer.shared_expert_device_output, workspace);
    inputs.clear();
    return combined;
}

static void norm_fwd(const CompiledModel& model,
                     TensorHandle weight,
                     const ActivationBuffer& input,
                     ActivationBuffer& output)
{
    if (model.descriptor.final_norm == NormType::LayerNorm)
    {
        forward_layer_norm(input,
                           model.weights.at(weight),
                           model.weights.at(model.final_norm_bias),
                           model.descriptor.norm_epsilon,
                           output);
        return;
    }
    forward_rms_norm(input,
                     model.weights.at(weight),
                     model.descriptor.norm_epsilon,
                     output,
                     model.descriptor.norm_weight_offset);
}

[[nodiscard]] static Result<void> hyper_head(const CompiledModel& model,
                                             const ActivationBuffer& input,
                                             HyperConnectionScratch& scratch,
                                             ActivationBuffer& output)
{
    if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
    {
        return hc_head(input,
                       model.weights.at(model.hyper_head_function),
                       model.weights.at(model.hyper_head_scale),
                       model.weights.at(model.hyper_head_base),
                       model.descriptor.hyper_connection_multiplier,
                       model.descriptor.norm_epsilon,
                       model.descriptor.hyper_connection_epsilon,
                       output,
                       scratch,
                       model.opt.optimization_flags);
    }
    if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
    {
        return gr_head(input,
                       model.weights,
                       model.operators,
                       model.gated_residual_head,
                       model.descriptor,
                       output,
                       scratch,
                       model.opt.optimization_flags);
    }
    return {};
}

[[nodiscard]] static Result<void> lm_head_linear(const CompiledModel& model,
                                                 const Layer& node,
                                                 TensorHandle weight,
                                                 const ActivationBuffer& input,
                                                 ActivationBuffer& logits)
{
    const TensorData& matrix = model.weights.at(weight);
    const CompiledOperator* op = model.operators.find_weight(weight);
    if (model.lm_head_bias != invalid_tensor_handle)
    {
        forward_linear(matrix,
                       model.weights.at(model.lm_head_bias),
                       input,
                       logits,
                       model.opt.optimization_flags,
                       op,
                       node.backend);
    }
    else
    {
        forward_linear(matrix,
                       input,
                       logits,
                       model.opt.optimization_flags,
                       op,
                       node.backend);
    }
    return {};
}

[[nodiscard]] static Result<void> finish_head(const std::shared_ptr<GpuSegment_vulkan>& head,
                                              GpuSegmentWorkspace_vulkan& gpu,
                                              ActivationBuffer& hidden,
                                              ActivationBuffer& logits,
                                              bool produce_logits,
                                              bool last_row_only,
                                              std::string_view error_prefix)
{
    std::string error;
    if (!head || !head->finish(produce_logits, last_row_only, hidden, logits, gpu, error))
        return Error{ErrorCode::InternalError, std::string(error_prefix) + error};
    return {};
}

[[nodiscard]] static Result<bool> publish_pending_head(std::shared_ptr<GpuSegmentContinuation_vulkan>& pending,
                                                       ExecutionNodeId& target,
                                                       ExecutionNodeId node_id,
                                                       ActivationBuffer& logits,
                                                       GpuSegmentWorkspace_vulkan& gpu,
                                                       std::shared_ptr<ResidentFfnRegion_vulkan>& region,
                                                       std::string_view error_prefix)
{
    if (!pending)
        return false;
    std::string error;
    if (target != node_id || !pending->publish_head(logits, gpu, error))
        return Error{ErrorCode::InternalError, std::string(error_prefix) + error};
    pending.reset();
    region.reset();
    target = invalid_execution_node_id;
    return true;
}

[[nodiscard]] static Result<bool> publish_segment(std::shared_ptr<GpuSegmentContinuation_vulkan>& pending,
                                                  ExecutionNodeId& target,
                                                  ExecutionNodeId node_id,
                                                  ExpertDispatchPlan& plan,
                                                  GpuSegmentWorkspace_vulkan& gpu,
                                                  std::shared_ptr<ResidentFfnRegion_vulkan>& region,
                                                  std::string_view error_prefix)
{
    if (!pending)
        return false;
    std::string error;
    if (target != node_id || !pending->publish(plan, gpu, error))
        return Error{ErrorCode::InternalError, std::string(error_prefix) + error};
    pending.reset();
    region.reset();
    target = invalid_execution_node_id;
    return true;
}

[[nodiscard]] static Result<void> fill_attn_rows(std::span<const DecodeBatchEntry> entries,
                                                 uint32_t layer_id,
                                                 std::vector<uint64_t>& positions,
                                                 std::vector<LayerCache*>& caches)
{
    positions.resize(entries.size());
    caches.resize(entries.size());
    for (size_t row = 0; row < entries.size(); ++row)
    {
        SessionState& state = *entries[row].state;
        if (layer_id >= state.layers.size())
            return Error{ErrorCode::InternalError, "attention layer cache is out of range"};
        positions[row] = entries[row].position_offset;
        caches[row] = &state.layers[layer_id];
    }
    return {};
}

[[nodiscard]] static Result<void> fwd_lm_head(const CompiledModel& model,
                                              const Layer& node,
                                              const CompiledOperator& lm_head_operator,
                                              const ActivationBuffer& input,
                                              ActivationBuffer& norm_output,
                                              ActivationBuffer& logits,
                                              bool norm_deferred)
{
    if (norm_deferred && node.backend == ExecutionBackend::Vulkan
        && lm_head_operator.bfloat16)
    {
        if (!lm_head_operator.bfloat16->forward_rms_norm_chain(input, logits))
            return Error{ErrorCode::InternalError, "Vulkan LM head RMSNorm chain execution failed"};
        return {};
    }
    const ActivationBuffer* project_input = &input;
    if (norm_deferred)
    {
        forward_rms_norm(input,
                         model.weights.at(node.weights[1]),
                         model.descriptor.norm_epsilon,
                         norm_output,
                         model.descriptor.norm_weight_offset);
        project_input = &norm_output;
    }
    return lm_head_linear(model, node, node.weights[0], *project_input, logits);
}

Result<void> forward_model(const CompiledModel& model,
                           std::span<const int32_t> input_ids,
                           SessionStatistics& statistics,
                           SessionState& state,
                           uint64_t position_offset,
                           LogitsOutput logits_output)
{
    const CpuOpenMpThreadLimitScope openmp_scope;
    const ScopedExpertBackendForeground backend_fg(model.expert_backend);
    if (attn_state_unknown(state))
    {
        return Error{
            ErrorCode::InternalError,
            "Session state is unavailable after a failed Vulkan Attention update"};
    }
    VulkanStatistics initial_vulkan_execution;
    if (model.vulkan_runtime)
        initial_vulkan_execution = get_vulkan_statistics(model.vulkan_runtime);
    for (int32_t token_id : input_ids)
    {
        if (token_id < 0 || static_cast<uint32_t>(token_id) >= model.descriptor.vocabulary_size)
            return Error{ErrorCode::InvalidArgument, "token id is outside the model vocabulary"};
    }

    if (input_ids.size() > std::numeric_limits<uint64_t>::max() - position_offset)
        return Error{ErrorCode::InvalidArgument, "execution position overflows"};
    auto output_size = ensure_graph(model, input_ids.size(), position_offset + input_ids.size());
    if (!output_size)
        return output_size.error();
    reset_state(model, state);
    state.gpu_segment.reserve_host_output(output_size.value());

    const bool use_speculative_context = model.speculative.enabled()
                                         && state.use_speculative_context;
    const bool needs_mtp_final_hidden = use_speculative_context
                                        && model.speculative.kind == SpeculativeModelKind::Mtp;
    ActivationBuffer& hidden = state.hidden;
    if (use_speculative_context)
    {
        const uint32_t speculative_hidden_columns = model.speculative.kind == SpeculativeModelKind::Mtp
                                                        ? model.descriptor.hidden_size
                                                        : model.descriptor.hidden_size
                                                              * static_cast<uint32_t>(model.speculative.target_layer_ids.size());
        state.speculative_main_hidden.reset(input_ids.size(),
                                            speculative_hidden_columns,
                                            true);
        state.speculative_main_hidden_position = position_offset;
        if (state.speculative_layers.size() != model.speculative.graph.layer_plans.size())
            state.speculative_layers.resize(model.speculative.graph.layer_plans.size());
        if (model.speculative.kind == SpeculativeModelKind::Mtp)
        {
            state.speculative_input_ids.assign(input_ids.begin(),
                                               input_ids.end());
            state.speculative_direct_alignment_ids.clear();
        }
    }
    else
    {
        state.speculative_main_hidden.clear();
    }
    if (logits_output == LogitsOutput::None)
        state.logits.clear();
    PendingRouterPrediction pending_predict;
    bool norm_deferred = false;
    bool logits_done = false;
    LatentLayerWorkspace_vulkan& layer_workspace = state.latent_layer;
    const ScopedLatentLayerWorkspace latent_workspace_scope(layer_workspace,
                                                            state.expert_scratch.combine_inputs);
    const ScopedGatedResidualWorkspace gated_residual_scope(state.gated_residual_layer);
    const ScopedGpuSegmentWorkspace gpu_workspace_scope(state.gpu_segment);
    bool resident_head_finished = false;
    if (model.gated_residual_chain && (use_speculative_context || has_flag(model.opt.flags, OptionRouterPrediction)))
        return Error{ErrorCode::InvalidArgument, "continuous gated residual Vulkan execution does not support speculative cache transactions or router prediction"};
    bool router_fused = false;
    bool head_deferred = false;
    const bool use_layer_fusion = !use_speculative_context
                                  && !has_flag(model.opt.flags, OptionRouterPrediction);
    const auto& gpu_head = model.operators.at_weight(model.hyper_head_function).layer_head;
    std::shared_ptr<GpuSegmentContinuation_vulkan> pending_continuation;
    std::shared_ptr<ResidentFfnRegion_vulkan> continuation_region;
    const ScopedResidentContinuation continuation_scope{pending_continuation, continuation_region};
    ExecutionNodeId continuation_target = invalid_execution_node_id;
    bool resident_chain_finished = false;
    uint32_t resident_prefix_layers = 0;

    const auto fwd_embedding = [&](const Layer* node) -> Result<void> {
        if (node->weights.size() != 1)
            return Error{ErrorCode::InternalError, "token embedding node has an invalid weight binding"};
        forward_embedding(model.weights.at(node->weights[0]),
                          input_ids,
                          hidden);
        hyper_connection_expand(hidden, model.descriptor.hyper_connection_multiplier, state.expert_scratch.staged_output);
        if (model.resident_submission_rows != 0)
        {
            if (use_speculative_context)
                return Error{ErrorCode::InternalError, "resident graph plan cannot consume speculative hidden captures"};
            auto& positions = state.attention_positions;
            positions.resize(input_ids.size());
            for (size_t row = 0; row < positions.size(); ++row) positions[row] = position_offset + row;
            auto& caches = state.attention_caches;
            caches.clear();
            auto& options = state.expert_scratch.bank_options;
            auto& bank_ids = state.expert_scratch.bank_ids;
            const size_t layer_count = model.graph.layer_plans.size();
            const uint32_t bank_layers = model.resident_bank_layers == 0
                                             ? static_cast<uint32_t>(layer_count)
                                             : model.resident_bank_layers;
            const bool partial = bank_layers < layer_count;
            options.clear();
            bank_ids.resize(layer_count);
            options.resize(layer_count);
            uint64_t assignments = 0;
            for (size_t index = 0; index < layer_count; ++index)
            {
                const auto& layer = model.graph.layer_plans[index];
                if (layer.layer_id >= state.layers.size())
                    return Error{ErrorCode::InternalError, "resident graph layer cache is out of range"};
                caches.insert(caches.end(), input_ids.size(), &state.layers[layer.layer_id]);
                if (index < bank_layers && !layer.moe.experts.empty())
                {
                    options[index] = route_opt(model, layer.moe, input_ids, bank_ids[index]);
                    assignments += input_ids.size() * layer.moe.top_k;
                }
            }
            std::string error;
            const bool completed = GpuSegment_vulkan::forward_resident(model, hidden, positions, caches, options,
                                                                       !partial && logits_output != LogitsOutput::None,
                                                                       !partial && logits_output == LogitsOutput::Last,
                                                                       state.logits, error, &state.gpu_segment,
                                                                       partial ? bank_layers : 0);
            if (!completed)
                return Error{ErrorCode::InternalError, "resident graph chain failed: " + error};
            statistics.expert_assignments += assignments;
            statistics.expert_gpu_routes += assignments;
            if (partial)
            {
                if (!GpuSegment_vulkan::materialize_hidden(state.gpu_segment, hidden, error))
                    return Error{ErrorCode::InternalError, "resident graph prefix boundary failed: " + error};
                resident_prefix_layers = bank_layers;
                return {};
            }
            logits_done = logits_output != LogitsOutput::None;
            resident_chain_finished = true;
        }
        return {};
    };
    const auto fwd_norm = [&](const Layer* node) -> Result<void> {
        auto published = publish_pending_head(pending_continuation, continuation_target, node->id,
                                              state.logits, state.gpu_segment, continuation_region,
                                              "resident Vulkan head continuation failed: ");
        if (!published)
            return published.error();
        if (published.value())
        {
            resident_head_finished = true;
            logits_done = logits_output != LogitsOutput::None;
            return {};
        }
        if (state.gpu_segment.active())
        {
            std::string error;
            if (model.gpu_head && !needs_mtp_final_hidden)
            {
                auto finished = finish_head(model.gpu_head, state.gpu_segment, hidden,
                                            state.logits, logits_output != LogitsOutput::None,
                                            logits_output == LogitsOutput::Last,
                                            "resident Vulkan head failed: ");
                if (!finished)
                    return finished.error();
                resident_head_finished = true;
                logits_done = logits_output != LogitsOutput::None;
                return {};
            }
            if (!GpuSegment_vulkan::materialize_hidden(state.gpu_segment, hidden, error))
                return Error{ErrorCode::InternalError, "resident Vulkan head boundary failed: " + error};
        }
        if (model.gated_residual_chain)
        {
            if (!GatedResidualLayer_vulkan::finish(model, logits_output != LogitsOutput::None,
                                                   logits_output == LogitsOutput::Last, state.logits, state.gated_residual_layer))
                return Error{ErrorCode::InternalError, "continuous gated residual Vulkan LM head execution failed"};
            logits_done = logits_output != LogitsOutput::None;
            return {};
        }
        const size_t weight_count = model.descriptor.final_norm == NormType::None ? 1 : 2;
        if (node->weights.size() != weight_count)
            return Error{ErrorCode::InternalError, "final norm node has an invalid weight binding"};
        if (model.descriptor.final_norm == NormType::RmsNorm
            && model.descriptor.hyper_connection_multiplier == 1
            && model.descriptor.hyper_connection_kind == HyperConnectionKind::None
            && !needs_mtp_final_hidden)
        {
            if (logits_output == LogitsOutput::None)
                return {};
            norm_deferred = !model.vulkan_runtime
                            && logits_output == LogitsOutput::Last
                            && hidden.rows() > 1;
            if (!norm_deferred && model.vulkan_runtime)
            {
                const CompiledOperator& lm_head_operator = model.operators.at_weight(node->weights.back());
                norm_deferred = lm_head_operator.bfloat16
                                && lm_head_operator.bfloat16->has_rms_norm_chain();
            }
            if (norm_deferred)
                return {};
        }
        if (layer_workspace.active())
        {
            if (gpu_head && LatentLayer_vulkan::can_finish(layer_workspace, *gpu_head))
            {
                head_deferred = true;
                return {};
            }
            if (!LatentLayer_vulkan::materialize_hidden(layer_workspace, hidden))
                return Error{ErrorCode::InternalError, "cannot materialize final latent hidden state"};
        }
        if (model.descriptor.hyper_connection_kind != HyperConnectionKind::None)
        {
            auto head = hyper_head(model, hidden, state.hyper_connection_scratch,
                                   state.expert_scratch.staged_output);
            if (!head)
                return head.error();
            hidden.swap(state.expert_scratch.staged_output);
        }
        if (model.descriptor.final_norm != NormType::None)
        {
            norm_fwd(model, node->weights[0], hidden,
                     state.expert_scratch.staged_output);
            hidden.swap(state.expert_scratch.staged_output);
        }
        if (needs_mtp_final_hidden)
        {
            for (size_t row = 0; row < hidden.rows(); ++row)
            {
                std::copy_n(hidden.row(row),
                            model.descriptor.hidden_size,
                            state.speculative_main_hidden.row(row));
            }
        }
        return {};
    };
    const auto fwd_head = [&](const Layer* node) -> Result<void> {
        if (resident_head_finished) return {};
        if (model.gated_residual_chain) return {};
        const size_t weight_count = model.descriptor.final_norm == NormType::None ? 1 : 2;
        if (node->weights.size() != weight_count)
            return Error{ErrorCode::InternalError, "LM head node has an invalid weight binding"};
        if (head_deferred)
        {
            if (!LatentLayer_vulkan::finish(layer_workspace, *gpu_head, logits_output != LogitsOutput::None,
                                            logits_output == LogitsOutput::Last, hidden, state.logits))
                return Error{ErrorCode::InternalError, "continuous latent LM head execution failed"};
            logits_done = logits_output != LogitsOutput::None;
            head_deferred = false;
            return {};
        }
        if (logits_output == LogitsOutput::None)
            return {};
        const CompiledOperator& lm_head_operator = model.operators.at_weight(node->weights[0]);
        const ActivationBuffer* lm_head_input = &hidden;
        if (logits_output == LogitsOutput::Last)
        {
            if (hidden.rows() == 0)
                return Error{ErrorCode::InternalError, "LM head has no final hidden row"};
            if (hidden.rows() > 1)
            {
                ActivationBuffer& last_hidden = state.lm_head_input;
                last_hidden.reset(1, hidden.columns(), false);
                std::copy_n(hidden.row(hidden.rows() - 1),
                            hidden.columns(),
                            last_hidden.row(0));
                lm_head_input = &last_hidden;
            }
        }
        auto headed = fwd_lm_head(model, *node, lm_head_operator, *lm_head_input,
                                  state.final_norm, state.logits, norm_deferred);
        if (!headed)
            return headed.error();
        norm_deferred = false;
        logits_done = true;
        return {};
    };
    const auto fwd_ffn = [&](const Layer* node, const CompiledLayerPlan& layer, LayerState& layer_state) -> Result<void> {
        return fwd_dense_layer(model, layer, hidden, layer_state, state.gpu_segment,
                               layer_workspace, node->backend);
    };
    const auto fwd_attn = [&](const Layer* node, const CompiledLayerPlan& layer, LayerState& layer_state, const MoeBlockPlan& moe) -> Result<void> {
        const auto segment = node->plan_index < model.gpu_segments.size() ? model.gpu_segments[node->plan_index] : nullptr;
        if (segment)
        {
            if (node->backend != ExecutionBackend::Vulkan)
                return Error{ErrorCode::InternalError, "resident graph segment has a non-Vulkan Attention node"};
            std::string error;
            auto published = publish_segment(pending_continuation, continuation_target, node->id,
                                             layer_state.dispatch_plan, state.gpu_segment,
                                             continuation_region,
                                             "resident Vulkan continuation at layer "
                                                 + std::to_string(layer.layer_id) + " failed: ");
            if (!published)
                return published.error();
            if (!published.value())
            {
                auto& positions = state.attention_positions;
                auto& caches = state.attention_caches;
                positions.resize(input_ids.size());
                caches.assign(input_ids.size(), &state.layers[layer.layer_id]);
                for (size_t row = 0; row < positions.size(); ++row) positions[row] = position_offset + row;
                auto options = route_opt(model, moe, input_ids, state.expert_scratch.explicit_expert_ids);
                if (!segment->forward(hidden, positions, caches, options, layer_state.dispatch_plan, state.gpu_segment, error))
                    return Error{ErrorCode::InternalError, "resident Vulkan segment at layer " + std::to_string(layer.layer_id) + " failed: " + error};
            }
            layer_state.normalized.reset(input_ids.size(), model.descriptor.hidden_size, false);
            layer_state.normalized_device = state.gpu_segment.device_input();
            layer_state.gpu_segment = true;
            layer_state.require_device_execution = true;
            router_fused = true;
            return {};
        }
        if (state.gpu_segment.active())
        {
            std::string error;
            if (!GpuSegment_vulkan::materialize_hidden(state.gpu_segment, hidden, error))
                return Error{ErrorCode::InternalError, "resident Vulkan graph boundary failed: " + error};
        }
        if (model.gated_residual_chain)
        {
            const auto& fused = model.operators.at_weight(layer.attention.output_weight).gated_residual_layer;
            auto options = route_opt(model, moe, input_ids, state.expert_scratch.explicit_expert_ids);
            std::string device_error;
            if (!fused || !fused->forward(hidden, input_ids, position_offset, state.layers[layer.layer_id], options, layer_state.dispatch_plan, state.gated_residual_layer, device_error))
                return Error{ErrorCode::InternalError, "continuous gated residual Vulkan layer " + std::to_string(layer.layer_id) + " execution failed: " + device_error};
            layer_state.normalized.reset(input_ids.size(), model.descriptor.hidden_size, false);
            layer_state.normalized_device = state.gated_residual_layer.device_input();
            layer_state.require_device_execution = true;
            router_fused = true;
            return {};
        }
        if (layer_workspace.active()
            && (layer.ple.enabled() || layer.attention.kind != AttentionKind::MultiHeadLatent)
            && !LatentLayer_vulkan::materialize_hidden(layer_workspace, hidden))
            return Error{ErrorCode::InternalError, "cannot materialize latent state for the next attention input"};
        if (layer.ple.enabled())
        {
            auto ple = forward_ple(model.weights,
                                   model.operators,
                                   layer.ple,
                                   model.descriptor,
                                   input_ids, state.layers[layer.layer_id], state.attention_scratch, hidden,
                                   model.opt.optimization_flags);
            if (!ple)
                return ple.error();
        }
        HyperConnectionMix& attn_mix = state.hyper_connection_scratch.transient_mix;
        const ActivationBuffer* attn_input = &hidden;
        if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
        {
            auto mixed = gr_pre(hidden,
                                model.weights,
                                model.operators,
                                layer.attention_gated_residual,
                                model.descriptor,
                                attn_mix,
                                state.hyper_connection_scratch,
                                model.opt.optimization_flags);
            if (!mixed)
                return mixed.error();
            attn_input = &attn_mix.reduced;
        }
        if (layer.attention.kind == AttentionKind::GatedDeltaNet)
        {
            auto gated_delta = forward_gated_delta(model.weights,
                                                   model.operators,
                                                   layer.attention,
                                                   node->backend,
                                                   state.layers[layer.layer_id],
                                                   state.gated_delta_scratch,
                                                   *attn_input,
                                                   state.gated_delta_scratch.output,
                                                   model.opt.optimization_flags);
            if (!gated_delta)
                return gated_delta.error();
            if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
            {
                auto connected = gr_post(state.gated_delta_scratch.output, hidden,
                                         attn_mix,
                                         model.descriptor.hyper_connection_multiplier,
                                         hidden);
                if (!connected)
                    return connected.error();
            }
            else
            {
                hidden.swap(state.gated_delta_scratch.output);
            }
        }
        else if (layer.attention.kind == AttentionKind::MultiHeadLatent)
        {
            const auto& fused = model.operators.at_weight(layer.attention.query_a_weight).latent_layer;
            if (use_layer_fusion && node->backend == ExecutionBackend::Vulkan && fused)
            {
                auto& positions = state.attention_positions;
                auto& caches = state.attention_caches;
                positions.resize(input_ids.size());
                caches.assign(input_ids.size(), &state.layers[layer.layer_id]);
                for (size_t row = 0; row < positions.size(); ++row)
                    positions[row] = position_offset + row;
                auto options = route_opt(model, moe, input_ids, state.expert_scratch.explicit_expert_ids);
                if (fused->forward(hidden, positions, caches, options, layer_state.normalized,
                                   layer_state.dispatch_plan, layer_workspace))
                {
                    layer_state.normalized_device = layer_workspace.device_input();
                    router_fused = true;
                    return {};
                }
                return Error{ErrorCode::InternalError, "Vulkan latent layer execution failed (layer "
                                                           + std::to_string(layer.layer_id) + ", device "
                                                           + std::to_string(layer.vulkan_device_index) + ")"};
            }
            if (layer_workspace.active() && !LatentLayer_vulkan::materialize_hidden(layer_workspace, hidden))
                return Error{ErrorCode::InternalError, "cannot materialize latent fallback state"};
            HyperConnectionMix& hyper_mix = state.hyper_connection_scratch.transient_mix;
            const ActivationBuffer* attention_input = &hidden;
            if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
            {
                auto mixed = hc_pre(hidden,
                                    model.weights.at(layer.hyper_connection.attention_function),
                                    model.weights.at(layer.hyper_connection.attention_scale),
                                    model.weights.at(layer.hyper_connection.attention_base),
                                    model.descriptor.hyper_connection_multiplier,
                                    model.descriptor.hyper_connection_iterations,
                                    model.descriptor.norm_epsilon,
                                    model.descriptor.hyper_connection_epsilon,
                                    hyper_mix,
                                    state.hyper_connection_scratch,
                                    model.opt.optimization_flags);
                if (!mixed)
                    return mixed.error();
                attention_input = &hyper_mix.reduced;
            }
            ActivationBuffer& attention_output = state.attention_scratch.output;
            auto output = forward_latent_attention(model.weights,
                                                   model.operators,
                                                   layer.attention,
                                                   node->backend,
                                                   position_offset,
                                                   state.layers[layer.layer_id],
                                                   state.attention_scratch,
                                                   *attention_input,
                                                   attention_output,
                                                   model.opt.optimization_flags);
            if (!output)
                return output.error();
            if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
            {
                auto connected = hc_post(attention_output, hidden, hyper_mix,
                                         model.descriptor.hyper_connection_multiplier,
                                         state.expert_scratch.staged_output);
                if (!connected)
                    return connected.error();
                hidden.swap(state.expert_scratch.staged_output);
            }
            else
            {
                add_batch_inplace(hidden, attention_output);
            }
        }
        else
        {
            auto attention = forward_attention(model.weights,
                                               model.operators,
                                               layer.attention,
                                               node->backend,
                                               position_offset,
                                               state.layers[layer.layer_id],
                                               state.attention_scratch,
                                               *attn_input,
                                               state.attention_scratch.output,
                                               model.opt.optimization_flags);
            if (!attention)
                return attention.error();
            if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
            {
                auto connected = gr_post(state.attention_scratch.output, hidden,
                                         attn_mix,
                                         model.descriptor.hyper_connection_multiplier,
                                         hidden);
                if (!connected)
                    return connected.error();
            }
            else
            {
                hidden.swap(state.attention_scratch.output);
            }
        }
        return {};
    };
    const auto fwd_router = [&](const CompiledLayerPlan& layer, LayerState& layer_state, const MoeBlockPlan& moe) -> Result<void> {
        if (router_fused)
        {
            return {};
        }
        layer_state.normalized_device.reset();
        auto completed_prediction = finish_predict(model,
                                                   layer.layer_id,
                                                   pending_predict,
                                                   state,
                                                   statistics,
                                                   hidden.rows());
        if (!completed_prediction)
            return completed_prediction.error();
        if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
        {
            auto mixed = gr_pre(hidden,
                                model.weights,
                                model.operators,
                                layer.ffn_gated_residual,
                                model.descriptor,
                                layer_state.ffn_hyper_mix,
                                state.hyper_connection_scratch,
                                model.opt.optimization_flags);
            if (!mixed)
                return mixed.error();
            // GatedResidual post needs only the injection coefficients.
            layer_state.normalized.swap(layer_state.ffn_hyper_mix.reduced);
            router_linear(model, moe, layer_state.normalized, layer_state.router_logits);
        }
        else
        {
            auto projected = router_fwd(model, layer, moe, hidden,
                                        layer_state.normalized, layer_state.router_logits,
                                        layer_state.ffn_hyper_mix,
                                        state.hyper_connection_scratch);
            if (!projected)
                return projected.error();
        }
        auto predicted = prefetch_routes(model,
                                         layer,
                                         layer_state.normalized,
                                         state,
                                         statistics,
                                         pending_predict);
        if (!predicted)
            return predicted.error();
        return {};
    };
    const auto fwd_dispatch = [&](const Layer* node, const CompiledLayerPlan& layer, LayerState& layer_state, const MoeBlockPlan& moe) -> Result<void> {
        ExpertDispatchOptions options = route_opt(model, moe, input_ids,
                                                  state.expert_scratch.explicit_expert_ids);
        if (!router_fused)
        {
            auto dispatched = forward_router(layer_state.router_logits.values(), static_cast<uint32_t>(layer_state.router_logits.rows()), options, layer_state.dispatch_plan);
            if (!dispatched)
                return dispatched.error();
        }

        ExpertDispatchPlan& plan = layer_state.dispatch_plan;
        if (hidden.rows() == 1 && layer.layer_id < state.layers.size())
            resolve_routes(model, layer, plan, state, statistics, true);
        prepare_experts(moe, layer_state, &statistics);
        if (has_flag(node->flags, ExecutionNodeRequestExperts))
        {
            auto requested = request_experts(model, moe, layer_state,
                                             state.expert_scratch,
                                             layer.layer_id);
            if (!requested)
                return requested.error();
        }
        return {};
    };
    const auto fwd_experts = [&](const Layer* node, const CompiledLayerPlan& layer, LayerState& layer_state, const MoeBlockPlan& moe) -> Result<void> {
        if (layer_state.experts_executed)
            return {};
        auto resident = open_region(model, layer_state, node->backend);
        if (!resident) return resident.error();
        if (layer_state.gpu_segment && layer_state.region)
        {
            if (pending_continuation)
                return Error{ErrorCode::InternalError, "resident Vulkan continuation reached another Expert group before its target"};
            const bool capture_boundary = use_speculative_context
                                          && std::find(model.speculative.target_layer_ids.begin(), model.speculative.target_layer_ids.end(), layer.layer_id)
                                                 != model.speculative.target_layer_ids.end();
            ExecutionNodeId target_id = invalid_execution_node_id;
            if (!capture_boundary)
            {
                uint32_t fence_id = invalid_execution_node_id;
                const auto resolved = GpuSegment_vulkan::continuation_target(model, node->id, fence_id);
                if (!resolved) return resolved.error();
                target_id = resolved.value();
            }
            if (target_id != invalid_execution_node_id && needs_mtp_final_hidden
                && model.graph.layers[target_id].kind == ExecutionNodeType::FinalNorm)
                target_id = invalid_execution_node_id;
            if (target_id != invalid_execution_node_id)
            {
                const auto& target = model.graph.layers[target_id];
                std::string error;
                if (target.kind == ExecutionNodeType::FinalNorm)
                    pending_continuation = model.gpu_head->prepare_head_continuation(logits_output != LogitsOutput::None,
                                                                                     logits_output == LogitsOutput::Last, state.gpu_segment, error);
                else
                {
                    const auto& next_layer = model.graph.layer_plans[target.plan_index];
                    if (next_layer.layer_id >= state.layers.size())
                        return Error{ErrorCode::InternalError, "resident Vulkan continuation layer cache is out of range"};
                    auto& positions = state.attention_positions;
                    auto& caches = state.attention_caches;
                    positions.resize(input_ids.size());
                    caches.assign(input_ids.size(), &state.layers[next_layer.layer_id]);
                    for (size_t row = 0; row < positions.size(); ++row) positions[row] = position_offset + row;
                    auto options = route_opt(model, next_layer.moe, input_ids, state.expert_scratch.explicit_expert_ids);
                    pending_continuation = model.gpu_segments[target.plan_index]->prepare_continuation(positions, caches, options,
                                                                                                       state.gpu_segment, error);
                }
                if (!pending_continuation || !layer_state.region->set_continuation(pending_continuation))
                    return Error{ErrorCode::InternalError, "cannot prepare resident Vulkan continuation from layer " + std::to_string(layer.layer_id) + ": " + error};
                continuation_region = layer_state.region;
                continuation_target = target_id;
            }
        }
        // Both host-normalized and resident inputs are independent of
        // routed weight supply; capability, rather than MLA, selects GPU work.
        const bool overlap = !layer_state.gpu_segment && moe.has_shared_expert
                             && node->backend == ExecutionBackend::Vulkan
                             && has_flag(model.opt.optimization_flags, OptimizationVulkanSharedExpertOverlap)
                             && support_vulkan_shared_experts(model.operators, moe);
        SharedOverlap shared{&model, &moe, &layer_state, router_fused, ExecutionBackend::Vulkan};
        auto executed = forward_moe(model,
                                    moe,
                                    layer_state,
                                    &statistics,
                                    state.expert_scratch,
                                    layer.layer_id,
                                    node->backend,
                                    has_flag(node->flags, ExecutionNodeCpuPrefetch),
                                    overlap ? MoEOverlap{&shared, run_shared_overlap} : MoEOverlap{});
        if (overlap && !shared.result) return shared.result.error();
        if (!executed)
            return executed.error();
        return {};
    };
    const auto fwd_combine = [&](const CompiledLayerPlan& layer, LayerState& layer_state, const MoeBlockPlan& moe) -> Result<void> {
        if (!layer_state.experts_executed)
        {
            return Error{ErrorCode::InternalError, "Combine executed before its Expert wave"};
        }
        if (layer_state.gpu_segment)
        {
            if (!pending_continuation && !GpuSegment_vulkan::defer_combine(layer_state.combined_device, state.gpu_segment))
                return Error{ErrorCode::InternalError, "resident Vulkan segment did not receive a complete Expert aggregate"};
            if (use_speculative_context)
            {
                const auto target = std::find(model.speculative.target_layer_ids.begin(), model.speculative.target_layer_ids.end(), layer.layer_id);
                if (target != model.speculative.target_layer_ids.end())
                {
                    std::string error;
                    if (!GpuSegment_vulkan::materialize_hidden(state.gpu_segment, hidden, error))
                        return Error{ErrorCode::InternalError, "resident speculative hidden capture failed: " + error};
                    capture_hidden(hidden, model.descriptor.hidden_size, model.descriptor.hyper_connection_multiplier,
                                   static_cast<size_t>(std::distance(model.speculative.target_layer_ids.begin(), target)), state.speculative_main_hidden);
                }
            }
            router_fused = false;
            layer_state.reset();
            return {};
        }
        if (moe.has_shared_expert && layer_state.shared_expert_output.rows() == 0
            && !layer_state.shared_expert_device_output)
            return Error{ErrorCode::InternalError, "Combine executed before Shared Expert group"};
        if (model.gated_residual_chain)
        {
            auto& inputs = state.expert_scratch.combine_inputs;
            inputs.clear();
            for (const auto& active : layer_state.active_experts())
            {
                if (!active.device_output)
                    return Error{ErrorCode::InternalError, "continuous gated residual Vulkan Expert output is not resident"};
                inputs.push_back({nullptr, active.device_output, active.batch.routes});
            }
            if ((moe.has_shared_expert && !layer_state.shared_expert_device_output)
                || !GatedResidualLayer_vulkan::defer_combine(inputs, input_ids.size(), model.descriptor.hidden_size,
                                                             layer_state.shared_expert_device_output, state.gated_residual_layer))
                return Error{ErrorCode::InternalError, "continuous gated residual Vulkan Expert combine failed"};
            inputs.clear();
            router_fused = false;
            layer_state.reset();
            return {};
        }
        bool device_combined = false;
        bool has_device_output = router_fused
                                 && static_cast<bool>(layer_state.shared_expert_device_output);
        if (router_fused)
        {
            for (const auto& active : layer_state.active_experts())
                has_device_output = has_device_output || static_cast<bool>(active.device_output);
        }
        if (router_fused && has_device_output)
        {
            device_combined = defer_experts(layer_state, state.expert_scratch, hidden.rows(),
                                            model.descriptor.hidden_size, layer_workspace);
            if (!device_combined)
            {
                return Error{ErrorCode::InternalError, "Vulkan latent Expert combine failed (layer "
                                                           + std::to_string(layer.layer_id) + ", device "
                                                           + std::to_string(layer.vulkan_device_index) + ")"};
            }
        }
        if (!device_combined)
        {
            ActivationBuffer& moe_output = layer_state.normalized;
            auto accumulated = combine_experts(model, moe, layer_state, state.expert_scratch,
                                               moe_output, hidden.rows());
            if (!accumulated)
                return accumulated.error();
            if (router_fused)
            {
                if (!LatentLayer_vulkan::defer_combine(moe_output, layer_workspace))
                    return Error{ErrorCode::InternalError, "cannot defer latent FFN post"};
                router_fused = false;
            }
            else if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
            {
                auto connected = gr_post(moe_output, hidden, layer_state.ffn_hyper_mix,
                                         model.descriptor.hyper_connection_multiplier,
                                         hidden);
                if (!connected)
                    return connected.error();
            }
            else if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
            {
                auto connected = hc_post(moe_output, hidden, layer_state.ffn_hyper_mix,
                                         model.descriptor.hyper_connection_multiplier,
                                         state.expert_scratch.staged_output);
                if (!connected)
                    return connected.error();
                hidden.swap(state.expert_scratch.staged_output);
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
        }
        if (device_combined) router_fused = false;

        if (use_speculative_context)
        {
            const auto target = std::find(model.speculative.target_layer_ids.begin(),
                                          model.speculative.target_layer_ids.end(),
                                          layer.layer_id);
            if (target != model.speculative.target_layer_ids.end())
            {
                capture_hidden(hidden,
                               model.descriptor.hidden_size,
                               model.descriptor.hyper_connection_multiplier,
                               static_cast<size_t>(std::distance(model.speculative.target_layer_ids.begin(), target)),
                               state.speculative_main_hidden);
            }
        }
        layer_state.reset();
        return {};
    };
    for (const ExecutionNodeId node_id : model.schedule.node_order)
    {
        if (node_id >= model.graph.layers.size())
            return Error{ErrorCode::InternalError, "execution schedule references an invalid node"};
        const Layer* node = &model.graph.layers[node_id];
        if (resident_chain_finished) continue;
        if (resident_prefix_layers != 0 && node->plan_index < resident_prefix_layers)
            continue;
        if (node->kind == ExecutionNodeType::TokenEmbedding)
        {
            auto stepped = fwd_embedding(node);
            if (!stepped) return stepped;
            continue;
        }
        if (node->kind == ExecutionNodeType::FinalNorm)
        {
            auto stepped = fwd_norm(node);
            if (!stepped) return stepped;
            continue;
        }
        if (node->kind == ExecutionNodeType::LmHead)
        {
            auto stepped = fwd_head(node);
            if (!stepped) return stepped;
            continue;
        }
        if (node->plan_index >= model.graph.layer_plans.size())
            return Error{ErrorCode::InternalError, "execution node layer is out of range"};

        const CompiledLayerPlan& layer = model.graph.layer_plans[node->plan_index];
        if (layer.layer_id >= state.layers.size())
            return Error{ErrorCode::InternalError, "execution layer id is out of range"};
        LayerState& layer_state = state.execution_state;
        const MoeBlockPlan& moe = layer.moe;

        Result<void> stepped;
        switch (node->kind)
        {
        case ExecutionNodeType::DenseFfn:
            stepped = fwd_ffn(node, layer, layer_state);
            break;
        case ExecutionNodeType::Attention:
            stepped = fwd_attn(node, layer, layer_state, moe);
            break;
        case ExecutionNodeType::Router:
            stepped = fwd_router(layer, layer_state, moe);
            break;
        case ExecutionNodeType::ExpertDispatch:
            stepped = fwd_dispatch(node, layer, layer_state, moe);
            break;
        case ExecutionNodeType::ExpertGroup:
            stepped = fwd_experts(node, layer, layer_state, moe);
            break;
        case ExecutionNodeType::SharedExpertGroup:
            stepped = fwd_shared_group(model, moe, layer_state.normalized.rows(), layer_state,
                                       router_fused, node->backend);
            break;
        case ExecutionNodeType::Combine:
            stepped = fwd_combine(layer, layer_state, moe);
            break;
        default:
            return Error{ErrorCode::InternalError, "unsupported execution graph node"};
        }
        if (!stepped) return stepped;
    }

    if (pending_continuation)
        return Error{ErrorCode::InternalError, "execution graph ended before its resident Vulkan continuation target"};
    if (pending_predict.result.valid())
    {
        return Error{
            ErrorCode::InternalError,
            "execution graph ended before a Router prediction target"};
    }

    if (logits_output != LogitsOutput::None && !logits_done)
        return Error{ErrorCode::InternalError, "execution graph did not produce logits"};
    if (logits_output == LogitsOutput::Last && state.logits.rows() != 1)
        return Error{ErrorCode::InternalError, "execution graph produced an invalid final logits row count"};
    if (model.vulkan_runtime)
    {
        const VulkanStatistics final_vulkan_execution = get_vulkan_statistics(model.vulkan_runtime);
        record_vulkan_delta(statistics,
                            initial_vulkan_execution,
                            final_vulkan_execution);
    }
    return {};
}

struct ScopedBatchWorkspace
{
    BatchWorkspace& workspace;

    ~ScopedBatchWorkspace()
    {
        ExpertScratch& expert = workspace.expert;
        workspace.staged_state.reset();

        expert.cache_requests.clear();
        expert.cache_leases.clear();
        expert.combine_inputs.clear();
        expert.backend_requests.clear();
        expert.decode_tasks.clear();
        expert.kernels.effective_tasks.clear();
        for (std::shared_ptr<const Mxfp4Q8PackedMatrix>& packed : expert.kernels.q8_down_packed)
            packed.reset();
        expert.backend_aggregated_output_valid = false;
        workspace.gated_delta_entries.clear();
        workspace.gated_delta_device_entries.clear();
        workspace.staged_attention_caches.clear();
        workspace.attention_batch_entries.clear();

        workspace.attention.latent_row_contexts.clear();
        workspace.attention.latent_projected_compressed_counts.clear();
        workspace.attention.latent_caches.clear();
    }
};

Result<void> forward_decode_batch(const CompiledModel& model,
                                  std::span<const DecodeBatchEntry> entries,
                                  BatchWorkspace& workspace)
{
    const ScopedExpertBackendForeground backend_fg(model.expert_backend);
    ScopedBatchWorkspace workspace_scope{workspace};
    workspace.expert.staged_logits_ready = false;
    workspace.expert.staged_logits_full_rows = false;
    if (entries.empty())
        return Error{ErrorCode::InvalidArgument, "decode batch cannot be empty"};
    uint64_t context_tokens = 0;
    for (const DecodeBatchEntry& entry : entries)
    {
        if (entry.position_offset == std::numeric_limits<uint64_t>::max())
            return Error{ErrorCode::InvalidArgument, "decode position overflows"};
        context_tokens = std::max(context_tokens, entry.position_offset + 1);
    }
    if (entries.size() > std::numeric_limits<uint32_t>::max())
        return Error{ErrorCode::InvalidArgument, "decode batch exceeds uint32 range"};
    auto output_size = ensure_graph(model, 1, context_tokens, static_cast<uint32_t>(entries.size()));
    if (!output_size)
        return output_size.error();
    workspace.gpu_segment.reserve_host_output(output_size.value());
    const CpuOpenMpThreadLimitScope openmp_scope;

    if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
    {
        for (const DecodeBatchEntry& entry : entries)
        {
            if (!entry.statistics || !entry.state)
                return Error{ErrorCode::InvalidArgument, "decode batch entry is incomplete"};
            if (entry.logits)
                entry.logits->clear();
            const std::array<int32_t, 1> input = {entry.input_id};
            auto executed = forward_model(model, input, *entry.statistics, *entry.state,
                                          entry.position_offset,
                                          entry.logits ? LogitsOutput::All : LogitsOutput::None);
            if (!executed)
                return executed.error();
            if (entry.logits)
            {
                if (entry.state->logits.rows() != 1
                    || entry.state->logits.columns() != model.descriptor.vocabulary_size)
                {
                    return Error{ErrorCode::InternalError, "gated-residual decode produced an invalid row count"};
                }
                const std::span<const float> values = entry.state->logits.values();
                entry.logits->assign(values.begin(), values.end());
            }
            else
            {
                if (entry.state->logits.rows() != 0)
                    return Error{ErrorCode::InternalError, "gated-residual decode produced unexpected logits"};
            }
        }
        return {};
    }

    const size_t session_count = entries.size();
    const size_t requested_count = static_cast<size_t>(std::count_if(entries.begin(), entries.end(), [](const DecodeBatchEntry& entry) {
        return entry.logits != nullptr;
    }));
    const bool all_output_logits = requested_count == session_count;
    for (const DecodeBatchEntry& entry : entries)
    {
        if (!entry.statistics || !entry.state)
            return Error{ErrorCode::InvalidArgument, "decode batch entry is incomplete"};
        if (entry.logits)
            entry.logits->clear();
        if (attn_state_unknown(*entry.state))
        {
            return Error{
                ErrorCode::InternalError,
                "staged Session state is unavailable after a failed Vulkan Attention update"};
        }
    }
    const uint32_t hyper_multiplier = model.descriptor.hyper_connection_multiplier;
    const uint32_t hyper_iterations = model.descriptor.hyper_connection_iterations;
    const float hyper_epsilon = model.descriptor.hyper_connection_epsilon;
    for (size_t session_index = 0; session_index < session_count; ++session_index)
    {
        const DecodeBatchEntry& entry = entries[session_index];
        if (entry.input_id < 0 || static_cast<uint32_t>(entry.input_id) >= model.descriptor.vocabulary_size)
        {
            return Error{ErrorCode::InvalidArgument, "token id is outside the model vocabulary"};
        }
        reset_state(model, *entry.state);
    }
    ExpertScratch& batch_scratch = workspace.expert;
    // Shared Vulkan resource windows cover each staged batch segment.
    VulkanStatistics batch_vulkan_before;
    if (model.vulkan_runtime)
        batch_vulkan_before = get_vulkan_statistics(model.vulkan_runtime);
    if (model.speculative.enabled())
    {
        const uint32_t speculative_hidden_columns = model.speculative.kind == SpeculativeModelKind::Mtp
                                                        ? model.descriptor.hidden_size
                                                        : model.descriptor.hidden_size
                                                              * static_cast<uint32_t>(model.speculative.target_layer_ids.size());
        for (const DecodeBatchEntry& entry : entries)
        {
            if (!entry.state->use_speculative_context)
                continue;
            entry.state->speculative_main_hidden.reset(1, speculative_hidden_columns, true);
            entry.state->speculative_main_hidden_position = entry.position_offset;
            if (entry.state->speculative_layers.size() != model.speculative.graph.layer_plans.size())
            {
                entry.state->speculative_layers.resize(model.speculative.graph.layer_plans.size());
            }
            if (model.speculative.kind == SpeculativeModelKind::Mtp)
            {
                entry.state->speculative_input_ids.assign(1,
                                                          entry.input_id);
                entry.state->speculative_direct_alignment_ids.clear();
            }
        }
    }

    auto merge_hidden_rows = [&entries, session_count](ActivationBuffer& merged) {
        const ActivationBuffer& first_hidden = entries.front().state->hidden;
        if (first_hidden.rows() != 1)
            return false;
        merged.reset(session_count, first_hidden.columns(), false);
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            const ActivationBuffer& hidden = entries[session_index].state->hidden;
            if (hidden.rows() != 1 || hidden.columns() != merged.columns())
            {
                return false;
            }
            std::copy_n(hidden.row(0),
                        merged.columns(),
                        merged.row(session_index));
        }
        return true;
    };
    auto split_hidden_rows = [&entries, session_count](const ActivationBuffer& merged) {
        if (merged.rows() != session_count)
            return false;
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            ActivationBuffer& hidden = entries[session_index].state->hidden;
            hidden.reset(1, merged.columns(), false);
            std::copy_n(merged.row(session_index),
                        merged.columns(),
                        hidden.row(0));
        }
        return true;
    };
    const bool use_speculative_context = std::any_of(entries.begin(), entries.end(), [](const DecodeBatchEntry& entry) {
        return entry.state->use_speculative_context;
    });
    bool norm_deferred = false;
    bool head_deferred = false;
    bool router_fused = false;
    LatentLayerWorkspace_vulkan& layer_workspace = workspace.latent_layer;
    const ScopedLatentLayerWorkspace latent_workspace_scope(layer_workspace,
                                                            workspace.expert.combine_inputs);
    const ScopedGpuSegmentWorkspace gpu_workspace_scope(workspace.gpu_segment);
    bool resident_router_ready = false;
    bool resident_head_finished = false;
    ExpertDispatchPlan& fused_routes = workspace.latent_routes;
    const auto& gpu_head = model.operators.at_weight(model.hyper_head_function).layer_head;
    const bool use_layer_fusion = (!model.speculative.enabled() || !use_speculative_context)
                                  && !has_flag(model.opt.flags, OptionRouterPrediction);
    std::shared_ptr<GpuSegmentContinuation_vulkan> pending_continuation;
    std::shared_ptr<ResidentFfnRegion_vulkan> continuation_region;
    const ScopedResidentContinuation continuation_scope{pending_continuation, continuation_region};
    ExecutionNodeId continuation_target = invalid_execution_node_id;
    bool resident_chain_finished = false;

    const auto fwd_staged_embedding = [&](const Layer* node) -> Result<void> {
        if (node->weights.size() != 1)
            return Error{ErrorCode::InternalError, "token embedding node has an invalid weight binding"};
        ExpertScratch& scratch = workspace.expert;
        std::vector<int32_t>& input_ids = workspace.staged_input_ids;
        input_ids.resize(session_count);
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            input_ids[session_index] = entries[session_index].input_id;
        }
        forward_embedding(model.weights.at(node->weights[0]),
                          input_ids,
                          scratch.staged_output);
        if (hyper_multiplier > 1)
        {
            hyper_connection_expand(scratch.staged_output, hyper_multiplier, scratch.staged_merged);
        }
        if (!split_hidden_rows(scratch.staged_output))
            return Error{ErrorCode::InternalError, "cannot split staged embedding rows"};
        if (model.resident_submission_rows != 0)
        {
            if (model.speculative.enabled() && use_speculative_context)
                return Error{ErrorCode::InternalError, "resident staged graph plan cannot consume speculative hidden captures"};
            auto& positions = workspace.staged_attention_positions;
            positions.resize(session_count);
            for (size_t row = 0; row < session_count; ++row) positions[row] = entries[row].position_offset;
            auto& caches = workspace.staged_attention_caches;
            caches.clear();
            auto& options = batch_scratch.bank_options;
            auto& bank_ids = batch_scratch.bank_ids;
            options.clear();
            bank_ids.resize(model.graph.layer_plans.size());
            options.resize(model.graph.layer_plans.size());
            uint64_t assignments = 0;
            for (size_t index = 0; index < model.graph.layer_plans.size(); ++index)
            {
                const auto& layer = model.graph.layer_plans[index];
                for (const auto& entry : entries)
                {
                    if (layer.layer_id >= entry.state->layers.size())
                        return Error{ErrorCode::InternalError, "resident staged graph cache is out of range"};
                    caches.push_back(&entry.state->layers[layer.layer_id]);
                }
                if (!layer.moe.experts.empty())
                {
                    options[index] = route_opt(model, layer.moe, input_ids, bank_ids[index]);
                    assignments += layer.moe.top_k;
                }
            }
            ActivationBuffer completed;
            std::string error;
            const bool executed = GpuSegment_vulkan::forward_resident(model, scratch.staged_output, positions, caches, options,
                                                                      requested_count != 0, false, completed, error, &workspace.gpu_segment);
            if (!executed)
                return Error{ErrorCode::InternalError, "resident staged graph chain failed: " + error};
            for (size_t row = 0; row < session_count; ++row)
            {
                if (entries[row].logits)
                    entries[row].logits->assign(completed.row(row), completed.row(row) + completed.columns());
                entries[row].statistics->expert_assignments += assignments;
                entries[row].statistics->expert_gpu_routes += assignments;
            }
            resident_chain_finished = true;
        }
        return {};
    };
    const auto fwd_staged_norm = [&](const Layer* node) -> Result<void> {
        if (pending_continuation)
        {
            ActivationBuffer& gpu_logits = workspace.expert.staged_output;
            auto published = publish_pending_head(pending_continuation, continuation_target, node->id,
                                                  gpu_logits, workspace.gpu_segment, continuation_region,
                                                  "resident staged Vulkan head continuation failed: ");
            if (!published)
                return published.error();
            if (published.value())
            {
                workspace.expert.staged_logits_ready = true;
                workspace.expert.staged_logits_full_rows = true;
                resident_head_finished = true;
                return {};
            }
        }
        if (workspace.gpu_segment.active())
        {
            std::string error;
            if (model.gpu_head && !(model.speculative.kind == SpeculativeModelKind::Mtp && use_speculative_context))
            {
                ActivationBuffer& gpu_logits = workspace.expert.staged_output;
                auto finished = finish_head(model.gpu_head, workspace.gpu_segment,
                                            workspace.expert.staged_merged, gpu_logits,
                                            requested_count != 0, false,
                                            "resident staged Vulkan head failed: ");
                if (!finished)
                    return finished.error();
                workspace.expert.staged_logits_ready = true;
                workspace.expert.staged_logits_full_rows = true;
                resident_head_finished = true;
                return {};
            }
            if (!GpuSegment_vulkan::materialize_hidden(workspace.gpu_segment, workspace.expert.staged_merged, error)
                || !split_hidden_rows(workspace.expert.staged_merged))
                return Error{ErrorCode::InternalError, "resident staged Vulkan head boundary failed: " + error};
        }
        const size_t weight_count = model.descriptor.final_norm == NormType::None ? 1 : 2;
        if (node->weights.size() != weight_count)
            return Error{ErrorCode::InternalError, "final norm node has an invalid weight binding"};
        const CompiledOperator& lm_head_operator = model.operators.at_weight(node->weights.back());
        if (model.descriptor.final_norm == NormType::RmsNorm
            && hyper_multiplier == 1
            && model.descriptor.hyper_connection_kind == HyperConnectionKind::None
            && !use_speculative_context
            && requested_count > 0
            && lm_head_operator.bfloat16
            && lm_head_operator.bfloat16->has_rms_norm_chain())
        {
            norm_deferred = true;
            return {};
        }
        if (layer_workspace.active())
        {
            if (gpu_head && LatentLayer_vulkan::can_finish(layer_workspace, *gpu_head)
                && (all_output_logits || requested_count == 0))
            {
                head_deferred = true;
                return {};
            }
            ActivationBuffer& materialized = workspace.expert.staged_output;
            if (gpu_head && LatentLayer_vulkan::can_finish(layer_workspace, *gpu_head))
            {
                if (!LatentLayer_vulkan::finish(layer_workspace, *gpu_head, false, false, materialized, workspace.staged_router_logits)
                    || !split_hidden_rows(materialized))
                    return Error{ErrorCode::InternalError, "cannot materialize staged latent head"};
                return {};
            }
            if (!LatentLayer_vulkan::materialize_hidden(layer_workspace, materialized) || !split_hidden_rows(materialized))
                return Error{ErrorCode::InternalError, "cannot materialize staged final latent state"};
        }
        ExpertScratch& scratch = workspace.expert;
        if (model.descriptor.hyper_connection_kind != HyperConnectionKind::None)
        {
            ActivationBuffer& merged_hyper = scratch.staged_merged;
            if (!merge_hidden_rows(merged_hyper))
            {
                return Error{
                    ErrorCode::InternalError,
                    "cannot merge staged hyper head rows"};
            }
            auto head = hyper_head(model, merged_hyper, workspace.hyper_connection,
                                   scratch.staged_output);
            if (!head)
                return head.error();
            if (!split_hidden_rows(scratch.staged_output))
                return Error{ErrorCode::InternalError, "cannot split staged hyper head rows"};
        }
        if (model.descriptor.final_norm != NormType::None)
        {
            ActivationBuffer& merged = scratch.staged_merged;
            if (!merge_hidden_rows(merged))
            {
                return Error{ErrorCode::InternalError, "cannot merge staged hidden rows"};
            }
            norm_fwd(model, node->weights[0], merged, scratch.staged_output);
            if (!split_hidden_rows(scratch.staged_output))
                return Error{ErrorCode::InternalError, "cannot split staged final norm rows"};
        }
        if (model.speculative.kind == SpeculativeModelKind::Mtp)
        {
            for (size_t session_index = 0;
                 session_index < session_count;
                 ++session_index)
            {
                SessionState& state = *entries[session_index].state;
                if (!state.use_speculative_context)
                    continue;
                std::copy_n(entries[session_index].state->hidden.row(0),
                            model.descriptor.hidden_size,
                            state.speculative_main_hidden.row(0));
            }
        }
        return {};
    };
    const auto fwd_staged_head = [&](const Layer* node) -> Result<void> {
        if (resident_head_finished) return {};
        const size_t weight_count = model.descriptor.final_norm == NormType::None ? 1 : 2;
        if (node->weights.size() != weight_count)
            return Error{ErrorCode::InternalError, "LM head node has an invalid weight binding"};
        if (head_deferred)
        {
            ActivationBuffer& materialized = workspace.expert.staged_merged;
            ActivationBuffer& gpu_logits = workspace.expert.staged_output;
            if (!LatentLayer_vulkan::finish(layer_workspace, *gpu_head, requested_count != 0, false, materialized, gpu_logits)
                || !split_hidden_rows(materialized))
                return Error{ErrorCode::InternalError, "continuous staged latent LM head execution failed"};
            workspace.expert.staged_logits_ready = true;
            workspace.expert.staged_logits_full_rows = true;
            head_deferred = false;
            return {};
        }
        if (requested_count == 0)
            return {};
        ExpertScratch& scratch = workspace.expert;
        ActivationBuffer& merged = scratch.staged_merged;
        const ActivationBuffer& first = entries.front().state->hidden;
        if (first.columns() == 0)
            return Error{ErrorCode::InternalError, "staged LM head rows have an invalid shape"};
        size_t requested_rows = 0;
        for (const DecodeBatchEntry& entry : entries)
        {
            if (entry.logits)
                ++requested_rows;
        }
        merged.reset(requested_rows, first.columns(), false);
        size_t output_row = 0;
        for (const DecodeBatchEntry& entry : entries)
        {
            if (!entry.logits)
                continue;
            const ActivationBuffer& hidden = entry.state->hidden;
            if (hidden.rows() != 1 || hidden.columns() != first.columns())
                return Error{ErrorCode::InternalError, "staged LM head rows have an invalid shape"};
            std::copy_n(hidden.row(0), hidden.columns(), merged.row(output_row++));
        }
        const CompiledOperator& lm_head_operator = model.operators.at_weight(node->weights[0]);
        auto headed = fwd_lm_head(model, *node, lm_head_operator, merged,
                                  workspace.staged_router_logits, scratch.staged_output,
                                  norm_deferred);
        if (!headed)
            return headed.error();
        norm_deferred = false;
        if (scratch.staged_output.rows() != requested_count)
        {
            return Error{ErrorCode::InternalError, "LM head produced an invalid row count"};
        }
        workspace.expert.staged_logits_ready = true;
        workspace.expert.staged_logits_full_rows = false;
        return {};
    };
    const auto fwd_staged_attn = [&](const Layer* node, const CompiledLayerPlan& layer, const MoeBlockPlan& moe) -> Result<void> {
        const auto segment = node->plan_index < model.gpu_segments.size() ? model.gpu_segments[node->plan_index] : nullptr;
        if (segment)
        {
            if (node->backend != ExecutionBackend::Vulkan)
                return Error{ErrorCode::InternalError, "resident staged segment has a non-Vulkan Attention node"};
            auto& initial_hidden = batch_scratch.staged_merged;
            if (!workspace.gpu_segment.active() && !merge_hidden_rows(initial_hidden))
                return Error{ErrorCode::InternalError, "cannot merge resident staged segment input"};
            std::string error;
            auto published = publish_segment(pending_continuation, continuation_target, node->id,
                                             fused_routes, workspace.gpu_segment, continuation_region,
                                             "resident staged Vulkan continuation at layer "
                                                 + std::to_string(layer.layer_id) + " failed: ");
            if (!published)
                return published.error();
            if (!published.value())
            {
                auto& positions = workspace.staged_attention_positions;
                auto& caches = workspace.staged_attention_caches;
                auto filled = fill_attn_rows(entries, layer.layer_id, positions, caches);
                if (!filled)
                    return filled.error();
                auto options = route_opt(model, moe, workspace.staged_input_ids, batch_scratch.explicit_expert_ids);
                if (!segment->forward(initial_hidden, positions, caches, options, fused_routes, workspace.gpu_segment, error))
                    return Error{ErrorCode::InternalError, "resident staged Vulkan segment at layer " + std::to_string(layer.layer_id) + " failed: " + error};
            }
            workspace.staged_state.normalized.reset(session_count, model.descriptor.hidden_size, false);
            router_fused = true;
            resident_router_ready = true;
            return {};
        }
        if (workspace.gpu_segment.active())
        {
            std::string error;
            if (!GpuSegment_vulkan::materialize_hidden(workspace.gpu_segment, batch_scratch.staged_merged, error)
                || !split_hidden_rows(batch_scratch.staged_merged))
                return Error{ErrorCode::InternalError, "resident staged Vulkan graph boundary failed: " + error};
        }
        if (layer_workspace.active() && layer.attention.kind != AttentionKind::MultiHeadLatent)
        {
            ActivationBuffer& materialized = workspace.expert.staged_merged;
            if (!LatentLayer_vulkan::materialize_hidden(layer_workspace, materialized) || !split_hidden_rows(materialized))
                return Error{ErrorCode::InternalError, "cannot materialize staged latent state for the next attention input"};
        }
        if (layer.attention.kind == AttentionKind::GatedDeltaNet)
        {
            std::vector<GatedDeltaBatchEntry>& gated_delta_entries = workspace.gated_delta_entries;
            gated_delta_entries.resize(session_count);
            for (size_t row = 0; row < session_count; ++row)
            {
                SessionState& state = *entries[row].state;
                if (layer.layer_id >= state.layers.size())
                    return Error{ErrorCode::InternalError, "gated delta layer cache is out of range"};
                gated_delta_entries[row] = {&state.hidden, &state.gated_delta_scratch,
                                            &state.layers[layer.layer_id],
                                            &state.gated_delta_scratch.output};
            }
            if (!forward_gated_delta_batch(model.weights,
                                           model.operators,
                                           layer.attention,
                                           node->backend,
                                           gated_delta_entries,
                                           workspace.gated_delta_device_entries,
                                           model.opt.optimization_flags))
            {
                return Error{
                    ErrorCode::InternalError,
                    "gated delta batch execution failed"};
            }
            for (size_t session_index = 0; session_index < session_count; ++session_index)
            {
                SessionState& state = *entries[session_index].state;
                state.hidden.swap(state.gated_delta_scratch.output);
            }
        }
        else if (layer.attention.kind == AttentionKind::MultiHeadLatent)
        {
            std::vector<uint64_t>& positions = workspace.staged_attention_positions;
            std::vector<LayerCache*>& caches = workspace.staged_attention_caches;
            positions.resize(session_count);
            caches.resize(session_count);
            ExpertScratch& scratch = batch_scratch;
            ActivationBuffer& merged_hidden = scratch.staged_merged;
            if (!merge_hidden_rows(merged_hidden))
            {
                return Error{
                    ErrorCode::InternalError,
                    "cannot merge staged attention rows"};
            }
            const auto& fused = model.operators.at_weight(layer.attention.query_a_weight).latent_layer;
            if (use_layer_fusion && node->backend == ExecutionBackend::Vulkan && fused)
            {
                auto filled = fill_attn_rows(entries, layer.layer_id, positions, caches);
                if (!filled)
                    return filled.error();
                auto options = route_opt(model, moe, workspace.staged_input_ids, batch_scratch.explicit_expert_ids);
                if (fused->forward(merged_hidden, positions, caches, options, workspace.staged_state.normalized,
                                   fused_routes, layer_workspace))
                {
                    router_fused = true;
                    return {};
                }
                return Error{ErrorCode::InternalError, "Vulkan latent batch execution failed (layer "
                                                           + std::to_string(layer.layer_id) + ", device "
                                                           + std::to_string(layer.vulkan_device_index) + ")"};
            }
            if (layer_workspace.active())
            {
                if (!LatentLayer_vulkan::materialize_hidden(layer_workspace, merged_hidden) || !split_hidden_rows(merged_hidden))
                    return Error{ErrorCode::InternalError, "cannot materialize staged latent fallback state"};
            }
            HyperConnectionScratch& hyper_scratch = workspace.hyper_connection;
            HyperConnectionMix& merged_mix = hyper_scratch.transient_mix;
            const ActivationBuffer* attention_input = &merged_hidden;
            if (hyper_multiplier > 1)
            {
                auto mixed = hc_pre(merged_hidden,
                                    model.weights.at(layer.hyper_connection.attention_function),
                                    model.weights.at(layer.hyper_connection.attention_scale),
                                    model.weights.at(layer.hyper_connection.attention_base),
                                    hyper_multiplier,
                                    hyper_iterations,
                                    model.descriptor.norm_epsilon,
                                    hyper_epsilon,
                                    merged_mix,
                                    hyper_scratch,
                                    model.opt.optimization_flags);
                if (!mixed)
                    return mixed.error();
                attention_input = &merged_mix.reduced;
            }
            auto filled = fill_attn_rows(entries, layer.layer_id, positions, caches);
            if (!filled)
                return filled.error();
            AttentionScratch& attention_scratch = workspace.attention;
            ActivationBuffer& attention_output = attention_scratch.output;
            auto merged_output = forward_latent_batch(model.weights, model.operators, layer.attention, node->backend, positions, caches,
                                                      attention_scratch, *attention_input, attention_output, model.opt.optimization_flags);
            if (!merged_output)
                return merged_output.error();
            if (hyper_multiplier > 1)
            {
                auto connected = hc_post(attention_output,
                                         merged_hidden,
                                         merged_mix,
                                         hyper_multiplier,
                                         batch_scratch.staged_output);
                if (!connected)
                    return connected.error();
                if (!split_hidden_rows(batch_scratch.staged_output))
                    return Error{ErrorCode::InternalError, "cannot split staged attention rows"};
            }
            else
            {
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    float* hidden = entries[session_index].state->hidden.row(0);
                    const float* attention = attention_output.row(session_index);
                    for (uint32_t column = 0; column < attention_output.columns(); ++column)
                        hidden[column] += attention[column];
                }
            }
        }
        else
        {
            std::vector<AttentionBatchEntry>& attention_entries = workspace.attention_batch_entries;
            attention_entries.resize(session_count);
            for (size_t row = 0; row < session_count; ++row)
            {
                SessionState& state = *entries[row].state;
                if (layer.layer_id >= state.layers.size())
                    return Error{ErrorCode::InternalError, "attention layer cache is out of range"};
                attention_entries[row] = {entries[row].position_offset,
                                          &state.layers[layer.layer_id],
                                          &state.attention_scratch, &state.hidden,
                                          &state.attention_scratch.output};
            }
            auto batched = forward_attention_batch(model.operators,
                                                   layer.attention,
                                                   node->backend,
                                                   attention_entries,
                                                   model.opt.optimization_flags);
            if (!batched)
                return batched.error();
            if (batched.value())
            {
                for (size_t session_index = 0;
                     session_index < session_count;
                     ++session_index)
                {
                    SessionState& state = *entries[session_index].state;
                    state.hidden.swap(state.attention_scratch.output);
                }
            }
            else
            {
                if (node->backend == ExecutionBackend::Vulkan
                    && model.vulkan_runtime)
                {
                    // Finish shared work before accounting for each Session separately.
                    const VulkanStatistics fallback_batch_vulkan_after = get_vulkan_statistics(model.vulkan_runtime);
                    for (const DecodeBatchEntry& entry : entries)
                    {
                        record_vulkan_delta(*entry.statistics,
                                            batch_vulkan_before,
                                            fallback_batch_vulkan_after);
                    }
                    batch_vulkan_before = fallback_batch_vulkan_after;
                }
                for (size_t session_index = 0;
                     session_index < session_count;
                     ++session_index)
                {
                    SessionState& state = *entries[session_index].state;
                    auto attention = forward_attention(model.weights,
                                                       model.operators,
                                                       layer.attention,
                                                       node->backend,
                                                       entries[session_index].position_offset,
                                                       state.layers[layer.layer_id],
                                                       state.attention_scratch,
                                                       state.hidden,
                                                       state.attention_scratch.output,
                                                       model.opt.optimization_flags);
                    if (!attention)
                    {
                        for (size_t affected_index = 0;
                             affected_index < session_count;
                             ++affected_index)
                        {
                            LayerCache& affected_cache = entries[affected_index].state->layers[layer.layer_id];
                            affected_cache.vulkan_attention_state_unknown = true;
                        }
                        return attention.error();
                    }
                    if (node->backend == ExecutionBackend::Vulkan
                        && model.vulkan_runtime)
                    {
                        const VulkanStatistics solo_vulkan_after = get_vulkan_statistics(model.vulkan_runtime);
                        record_vulkan_delta(*entries[session_index].statistics,
                                            batch_vulkan_before,
                                            solo_vulkan_after);
                        batch_vulkan_before = solo_vulkan_after;
                    }
                    state.hidden.swap(state.attention_scratch.output);
                }
            }
        }
        return {};
    };
    const auto fwd_staged_dispatch = [&](const Layer* node, const CompiledLayerPlan& layer, const MoeBlockPlan& moe) -> Result<void> {
        LayerState& combined = workspace.staged_state;
        combined.reset();
        if (resident_router_ready)
        {
            combined.normalized_device = workspace.gpu_segment.device_input();
            combined.gpu_segment = true;
            combined.require_device_execution = true;
        }
        else if (router_fused)
        {
            combined.normalized_device = layer_workspace.device_input();
        }
        if (combined.normalized.rows() != session_count
            || combined.normalized.columns() != model.descriptor.hidden_size)
        {
            return Error{ErrorCode::InternalError, "Expert group has no staged Router input"};
        }

        ExpertDispatchPlan& plan = combined.dispatch_plan;
        if (router_fused)
        {
            plan.assignment_count = fused_routes.assignment_count;
            plan.batches = std::move(fused_routes.batches);
        }
        else
        {
            plan.assignment_count = 0;
            plan.batches.clear();
            auto options = route_opt(model, moe, workspace.staged_input_ids,
                                     batch_scratch.explicit_expert_ids);
            auto dispatched = forward_router(workspace.staged_router_logits.values(),
                                             static_cast<uint32_t>(session_count),
                                             options, plan);
            if (!dispatched)
                return dispatched.error();
        }

        // Account the merged plan per Session once; the same plan then goes to
        // the shared Expert executor. No per-Session dispatch plan is built.
        const size_t expert_count = moe.experts.size();
        std::vector<uint8_t>& seen = batch_scratch.expert_seen;
        seen.assign(expert_count * session_count, 0);
        for (const ExpertBatch& batch : plan.batches)
        {
            const uint64_t weight_size = moe.experts[batch.expert_id].weight_size;
            for (const ExpertRoute& route : batch.routes)
            {
                if (route.token_index >= session_count)
                    return Error{ErrorCode::InternalError, "staged Expert route references an invalid Session"};
                SessionStatistics& statistics = *entries[route.token_index].statistics;
                ++statistics.expert_assignments;
                if (statistics.expert_route_weight_bytes
                    > std::numeric_limits<uint64_t>::max() - weight_size)
                {
                    statistics.expert_route_weight_bytes = std::numeric_limits<uint64_t>::max();
                }
                else
                {
                    statistics.expert_route_weight_bytes += weight_size;
                }
                uint8_t& slot = seen[batch.expert_id * session_count + route.token_index];
                if (slot != 0)
                    continue;
                slot = 1;
                if (statistics.expert_batch_weight_bytes
                    > std::numeric_limits<uint64_t>::max() - weight_size)
                {
                    statistics.expert_batch_weight_bytes = std::numeric_limits<uint64_t>::max();
                }
                else
                {
                    statistics.expert_batch_weight_bytes += weight_size;
                }
            }
        }

        prepare_experts(moe, combined, nullptr);
        // Keep Expert execution in canonical expert-id order so backend
        // publications and retained device outputs are deterministic.
        const auto active = combined.active_experts();
        std::sort(active.begin(), active.end(), [](const ExpertState& left, const ExpertState& right) {
            return left.batch.expert_id < right.batch.expert_id;
        });
        if (has_flag(node->flags, ExecutionNodeRequestExperts))
        {
            auto requested = request_experts(model, moe, combined, batch_scratch,
                                             layer.layer_id);
            if (!requested)
                return requested.error();
        }
        return {};
    };
    const auto fwd_staged_experts = [&](const Layer* node, const CompiledLayerPlan& layer, const MoeBlockPlan& moe) -> Result<void> {
        LayerState& combined = workspace.staged_state;
        auto resident = open_region(model, combined, node->backend);
        if (!resident) return resident.error();
        if (combined.gpu_segment && combined.region)
        {
            if (pending_continuation)
                return Error{ErrorCode::InternalError, "resident staged Vulkan continuation reached another Expert group before its target"};
            const bool capture_boundary = use_speculative_context
                                          && std::find(model.speculative.target_layer_ids.begin(), model.speculative.target_layer_ids.end(), layer.layer_id)
                                                 != model.speculative.target_layer_ids.end();
            ExecutionNodeId target_id = invalid_execution_node_id;
            if (!capture_boundary)
            {
                uint32_t fence_id = invalid_execution_node_id;
                const auto resolved = GpuSegment_vulkan::continuation_target(model, node->id, fence_id);
                if (!resolved) return resolved.error();
                target_id = resolved.value();
            }
            if (target_id != invalid_execution_node_id && use_speculative_context
                && model.speculative.kind == SpeculativeModelKind::Mtp
                && model.graph.layers[target_id].kind == ExecutionNodeType::FinalNorm)
                target_id = invalid_execution_node_id;
            if (target_id != invalid_execution_node_id)
            {
                const auto& target = model.graph.layers[target_id];
                std::string error;
                if (target.kind == ExecutionNodeType::FinalNorm)
                    pending_continuation = model.gpu_head->prepare_head_continuation(requested_count != 0, false,
                                                                                     workspace.gpu_segment, error);
                else
                {
                    const auto& next_layer = model.graph.layer_plans[target.plan_index];
                    auto& positions = workspace.staged_attention_positions;
                    auto& caches = workspace.staged_attention_caches;
                    auto filled = fill_attn_rows(entries, next_layer.layer_id, positions, caches);
                    if (!filled)
                        return filled.error();
                    auto options = route_opt(model, next_layer.moe, workspace.staged_input_ids, batch_scratch.explicit_expert_ids);
                    pending_continuation = model.gpu_segments[target.plan_index]->prepare_continuation(positions, caches, options,
                                                                                                       workspace.gpu_segment, error);
                }
                if (!pending_continuation || !combined.region->set_continuation(pending_continuation))
                    return Error{ErrorCode::InternalError, "cannot prepare resident staged Vulkan continuation from layer " + std::to_string(layer.layer_id) + ": " + error};
                continuation_region = combined.region;
                continuation_target = target_id;
            }
        }
        const bool overlap = !combined.gpu_segment && moe.has_shared_expert
                             && node->backend == ExecutionBackend::Vulkan
                             && has_flag(model.opt.optimization_flags, OptimizationVulkanSharedExpertOverlap)
                             && support_vulkan_shared_experts(model.operators, moe);
        SharedOverlap shared{&model, &moe, &combined, router_fused, ExecutionBackend::Vulkan};
        auto executed = forward_moe(model,
                                    moe,
                                    combined,
                                    nullptr,
                                    workspace.expert,
                                    layer.layer_id,
                                    node->backend,
                                    has_flag(node->flags, ExecutionNodeCpuPrefetch),
                                    overlap ? MoEOverlap{&shared, run_shared_overlap} : MoEOverlap{});
        if (overlap && !shared.result) return shared.result.error();
        if (!executed)
        {
            combined.reset();
            return executed.error();
        }
        for (ExpertState& active : combined.active_experts())
            active.lease = {};

        for (const ExpertState& active : combined.active_experts())
        {
            for (const ExpertRoute& route : active.batch.routes)
            {
                if (route.rank >= maximum_expert_route_ranks)
                    continue;
                if (route.token_index >= session_count)
                    return Error{ErrorCode::InternalError, "combined Expert route references an invalid session"};
                if (!has_flag(model.opt.flags, OptionRouterPrediction)
                    || !has_flag(model.opt.flags, OptionRankAdaptivePrefetch))
                    continue;
                SessionStatistics& route_statistics = *entries[route.token_index].statistics;
                ++route_statistics.expert_route_rank_demands[route.rank];
                route_statistics.expert_route_rank_demand_queue_time_microseconds[route.rank] += active.cache_wait_us;
            }
        }

        for (size_t index = 0; index < combined.active_experts().size(); ++index)
        {
            const ExpertState& active = combined.active_experts()[index];
            const bool published = index < batch_scratch.backend_status.size()
                                   && batch_scratch.backend_status[index] == ExpertStatus::Executed;
            if (!published && node->backend == ExecutionBackend::Vulkan
                && model.opt.hybrid_mode == HybridMode::HybridExperts)
            {
                return Error{ErrorCode::InternalError, "Vulkan Expert route was not published (layer "
                                                           + std::to_string(layer.layer_id) + ", op ExpertGroup, Vulkan device "
                                                           + std::to_string(layer.vulkan_device_index) + ")"};
            }
            std::vector<uint8_t>& touched = batch_scratch.expert_seen;
            touched.assign(session_count, 0);
            for (const ExpertRoute& route : active.batch.routes)
            {
                if (route.token_index >= session_count)
                    return Error{ErrorCode::InternalError, "combined Expert route references an invalid session"};
                SessionStatistics& statistics = *entries[route.token_index].statistics;
                if (touched[route.token_index] == 0)
                {
                    touched[route.token_index] = 1;
                    ++statistics.expert_batches;
                    if (!published)
                    {
                        ++statistics.expert_cpu_batches;
                        statistics.expert_cpu_fallbacks += node->backend == ExecutionBackend::Vulkan ? 1 : 0;
                    }
                }
                if (published)
                    ++statistics.expert_gpu_routes;
                else
                    ++statistics.expert_cpu_routes;
            }
        }
        combined.experts_executed = true;
        return {};
    };
    const auto fwd_staged_combine = [&](const CompiledLayerPlan& layer, const MoeBlockPlan& moe) -> Result<void> {
        LayerState& combined = workspace.staged_state;
        const bool capture_speculative_context = model.speculative.enabled() && use_speculative_context;
        const auto target = capture_speculative_context
                                ? std::find(model.speculative.target_layer_ids.begin(),
                                            model.speculative.target_layer_ids.end(),
                                            layer.layer_id)
                                : model.speculative.target_layer_ids.end();
        const size_t target_index = capture_speculative_context
                                        ? static_cast<size_t>(std::distance(model.speculative.target_layer_ids.begin(), target))
                                        : 0;

        if (combined.gpu_segment)
        {
            if (!combined.experts_executed
                || (!pending_continuation && !GpuSegment_vulkan::defer_combine(combined.combined_device, workspace.gpu_segment)))
            {
                return Error{ErrorCode::InternalError, "resident staged segment did not receive a complete Expert aggregate"};
            }
            if (target != model.speculative.target_layer_ids.end() && use_speculative_context)
            {
                std::string error;
                if (!GpuSegment_vulkan::materialize_hidden(workspace.gpu_segment, batch_scratch.staged_merged, error)
                    || !split_hidden_rows(batch_scratch.staged_merged))
                {
                    return Error{ErrorCode::InternalError, "resident staged speculative hidden capture failed: " + error};
                }
                for (const auto& entry : entries)
                {
                    if (entry.state->use_speculative_context)
                    {
                        capture_hidden(entry.state->hidden, model.descriptor.hidden_size, model.descriptor.hyper_connection_multiplier,
                                       target_index, entry.state->speculative_main_hidden);
                    }
                }
            }
            combined.reset();
            resident_router_ready = false;
            router_fused = false;
            return {};
        }

        if (!combined.experts_executed)
            return Error{ErrorCode::InternalError, "Combine executed before its Expert wave"};
        if (moe.has_shared_expert
            && combined.shared_expert_output.rows() != session_count
            && !combined.shared_expert_device_output)
        {
            return Error{ErrorCode::InternalError, "Combine executed before Shared Expert group"};
        }

        bool device_combined = false;
        bool has_device_output = router_fused
                                 && static_cast<bool>(combined.shared_expert_device_output);
        if (router_fused)
        {
            for (const ExpertState& active : combined.active_experts())
                has_device_output = has_device_output || static_cast<bool>(active.device_output);
        }
        if (router_fused && has_device_output)
        {
            device_combined = defer_experts(combined, batch_scratch, session_count,
                                            model.descriptor.hidden_size, layer_workspace);
            if (!device_combined)
            {
                return Error{ErrorCode::InternalError, "Vulkan latent batch Expert combine failed (layer "
                                                           + std::to_string(layer.layer_id) + ", device "
                                                           + std::to_string(layer.vulkan_device_index) + ")"};
            }
            router_fused = false;
        }

        if (!device_combined)
        {
            ActivationBuffer& moe_output = combined.normalized;
            auto accumulated = combine_experts(model, moe, combined, batch_scratch,
                                               moe_output, session_count);
            if (!accumulated)
                return accumulated.error();

            if (router_fused)
            {
                if (!LatentLayer_vulkan::defer_combine(moe_output, layer_workspace))
                    return Error{ErrorCode::InternalError, "cannot defer staged latent FFN post"};
                router_fused = false;
                device_combined = true;
            }
            else if (hyper_multiplier > 1)
            {
                if (!merge_hidden_rows(batch_scratch.staged_merged))
                    return Error{ErrorCode::InternalError, "cannot merge staged FFN residual rows"};
                auto connected = hc_post(moe_output,
                                         batch_scratch.staged_merged,
                                         combined.ffn_hyper_mix,
                                         hyper_multiplier,
                                         batch_scratch.staged_output);
                if (!connected)
                    return connected.error();
                if (!split_hidden_rows(batch_scratch.staged_output))
                    return Error{ErrorCode::InternalError, "cannot split staged hyper FFN rows"};
            }
            else
            {
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    float* hidden = entries[session_index].state->hidden.row(0);
                    const float* output = moe_output.row(session_index);
                    for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
                        hidden[column] += output[column];
                }
            }
        }
        else
        {
            batch_scratch.backend_aggregated_output_valid = false;
        }

        if (target != model.speculative.target_layer_ids.end())
        {
            if (device_combined)
            {
                if (!LatentLayer_vulkan::materialize_hidden(layer_workspace, batch_scratch.staged_merged)
                    || !split_hidden_rows(batch_scratch.staged_merged))
                {
                    return Error{ErrorCode::InternalError, "cannot materialize staged speculative hidden"};
                }
            }
            for (const auto& entry : entries)
            {
                if (entry.state->use_speculative_context)
                {
                    capture_hidden(entry.state->hidden, model.descriptor.hidden_size, hyper_multiplier,
                                   target_index, entry.state->speculative_main_hidden);
                }
            }
        }

        combined.reset();
        resident_router_ready = false;
        router_fused = false;
        return {};
    };
    for (const ExecutionNodeId node_id : model.schedule.node_order)
    {
        if (node_id >= model.graph.layers.size())
        {
            return Error{ErrorCode::InternalError, "execution schedule references an invalid node"};
        }
        const Layer* node = &model.graph.layers[node_id];
        if (resident_chain_finished) continue;
        if (node->kind == ExecutionNodeType::TokenEmbedding)
        {
            auto stepped = fwd_staged_embedding(node);
            if (!stepped) return stepped;
            continue;
        }
        if (node->kind == ExecutionNodeType::FinalNorm)
        {
            auto stepped = fwd_staged_norm(node);
            if (!stepped) return stepped;
            continue;
        }
        if (node->kind == ExecutionNodeType::LmHead)
        {
            auto stepped = fwd_staged_head(node);
            if (!stepped) return stepped;
            continue;
        }
        if (node->plan_index >= model.graph.layer_plans.size())
        {
            return Error{ErrorCode::InternalError, "execution node layer is out of range"};
        }

        const CompiledLayerPlan& layer = model.graph.layer_plans[node->plan_index];
        const MoeBlockPlan& moe = layer.moe;

        Result<void> stepped;
        switch (node->kind)
        {
        case ExecutionNodeType::DenseFfn:
        {
            ActivationBuffer& merged = batch_scratch.staged_merged;
            if (!workspace.gpu_segment.active() && !layer_workspace.active() && !merge_hidden_rows(merged))
            {
                stepped = Error{ErrorCode::InternalError, "cannot merge staged dense FFN rows"};
                break;
            }
            stepped = fwd_dense_layer(model, layer, merged, workspace.staged_state,
                                      workspace.gpu_segment, layer_workspace, node->backend);
            if (stepped && !split_hidden_rows(merged))
                stepped = Error{ErrorCode::InternalError, "cannot split staged dense FFN rows"};
            break;
        }
        case ExecutionNodeType::Attention:
            stepped = fwd_staged_attn(node, layer, moe);
            break;
        case ExecutionNodeType::Router:
        {
            if (!router_fused)
            {
                LayerState& combined = workspace.staged_state;
                ActivationBuffer& merged_hyper = batch_scratch.staged_output;
                if (!merge_hidden_rows(merged_hyper))
                {
                    stepped = Error{ErrorCode::InternalError, "cannot merge staged FFN hyper rows"};
                    break;
                }
                stepped = router_fwd(model, layer, moe, merged_hyper,
                                     combined.normalized, workspace.staged_router_logits,
                                     combined.ffn_hyper_mix, workspace.hyper_connection);
            }
            break;
        }
        case ExecutionNodeType::ExpertDispatch:
            stepped = fwd_staged_dispatch(node, layer, moe);
            break;
        case ExecutionNodeType::ExpertGroup:
            stepped = fwd_staged_experts(node, layer, moe);
            break;
        case ExecutionNodeType::SharedExpertGroup:
            stepped = fwd_shared_group(model, moe, session_count, workspace.staged_state,
                                       router_fused, node->backend);
            break;
        case ExecutionNodeType::Combine:
            stepped = fwd_staged_combine(layer, moe);
            break;
        default:
            return Error{ErrorCode::InternalError, "unsupported execution graph node"};
        }
        if (!stepped) return stepped;
    }

    if (pending_continuation)
        return Error{ErrorCode::InternalError, "staged execution graph ended before its resident Vulkan continuation target"};
    if (workspace.expert.staged_logits_ready)
    {
        size_t source_row = 0;
        for (size_t row = 0; row < session_count; ++row)
        {
            if (!entries[row].logits)
                continue;
            const size_t source = workspace.expert.staged_logits_full_rows ? row : source_row++;
            if (source >= workspace.expert.staged_output.rows())
                break;
            entries[row].logits->assign(workspace.expert.staged_output.row(source),
                                        workspace.expert.staged_output.row(source)
                                            + workspace.expert.staged_output.columns());
        }
    }
    for (size_t session_index = 0; session_index < session_count; ++session_index)
    {
        if (entries[session_index].logits
            && entries[session_index].logits->size() != model.descriptor.vocabulary_size)
        {
            return Error{ErrorCode::InternalError, "staged execution graph did not produce logits"};
        }
    }
    if (model.vulkan_runtime)
    {
        const VulkanStatistics batch_vulkan_after = get_vulkan_statistics(model.vulkan_runtime);
        for (const DecodeBatchEntry& entry : entries)
            record_vulkan_delta(*entry.statistics, batch_vulkan_before, batch_vulkan_after);
    }
    return {};
}

static void rank_experts(const ExpertDispatchPlan& plan,
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

static void set_prefetch(LayerCache& cache,
                         uint32_t target_top_k)
{
    if (cache.router_target_top_k == target_top_k)
        return;
    cache.router_target_top_k = target_top_k;
    cache.router_prefetch_width = std::min(2u, target_top_k);
    cache.router_decisions = 0;
    cache.router_last_adjustment = 0;
}

static void adapt_prefetch(LayerCache& cache,
                           const SessionStatistics& statistics,
                           bool adaptive)
{
    if (!adaptive)
    {
        cache.router_prefetch_width = cache.router_target_top_k;
        return;
    }
    if (cache.router_prefetch_width == 0)
        cache.router_prefetch_width = std::min(2u, cache.router_target_top_k);
    ++cache.router_decisions;
    if (cache.router_decisions % 16 != 0)
        return;

    if (cache.router_prefetch_width > 1)
    {
        const uint32_t marginal_rank = cache.router_prefetch_width - 1;
        const uint64_t predictions = statistics.expert_route_rank_predictions[marginal_rank];
        const uint64_t matches = statistics.expert_route_rank_matches[marginal_rank];
        if (predictions >= 8 && matches * 3 < predictions)
        {
            --cache.router_prefetch_width;
            cache.router_last_adjustment = cache.router_decisions;
            return;
        }
    }
    if (cache.router_decisions - cache.router_last_adjustment < 64)
        return;
    if (cache.router_prefetch_width < cache.router_target_top_k)
    {
        const uint32_t widest_prefetched_rank = cache.router_prefetch_width - 1;
        const uint64_t predictions = statistics.expert_route_rank_predictions[widest_prefetched_rank];
        const uint64_t matches = statistics.expert_route_rank_matches[widest_prefetched_rank];
        const uint32_t next_rank = cache.router_prefetch_width;
        const uint64_t demands = statistics.expert_route_rank_demands[next_rank];
        const uint64_t queued_microseconds = statistics.expert_route_rank_demand_queue_time_microseconds[next_rank];
        if (predictions >= 8
            && matches * 2 >= predictions
            && demands >= 8
            && queued_microseconds / demands >= 2000)
        {
            ++cache.router_prefetch_width;
            cache.router_last_adjustment = cache.router_decisions;
        }
    }
}

static void resolve_routes(const CompiledModel& model,
                           const CompiledLayerPlan& layer,
                           const ExpertDispatchPlan& plan,
                           SessionState& state,
                           SessionStatistics& statistics,
                           bool resolve_unused_predictions)
{
    if (!model.expert_cache
        || layer.layer_id >= state.layers.size()
        || !has_flag(model.opt.flags, OptionRouterPrediction))
    {
        return;
    }

    const uint32_t invalid_expert = std::numeric_limits<uint32_t>::max();
    LayerCache& cache = state.layers[layer.layer_id];
    std::array<uint32_t, maximum_expert_route_ranks> actual_storage;
    std::vector<uint32_t> actual_fallback;
    std::span<const uint32_t> actual_experts;
    if (layer.moe.top_k <= actual_storage.size())
    {
        const std::span<uint32_t> ranked = std::span<uint32_t>(actual_storage).first(layer.moe.top_k);
        rank_experts(plan, ranked);
        actual_experts = ranked;
    }
    else
    {
        actual_fallback.resize(layer.moe.top_k);
        rank_experts(plan, actual_fallback);
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
    if (resolve_unused_predictions && model.opt.num_concurrent_sessions == 1)
    {
        model.expert_cache->resolve_predictions(layer.layer_id,
                                                std::span<const std::string_view>(demanded_keys.data(), demanded_key_count));
    }

    cache.predicted_expert_ids.clear();
}

static const CompiledLayerPlan* next_predict(const CompiledModel& model,
                                             const CompiledLayerPlan& layer,
                                             const ActivationBuffer& router_input,
                                             SessionState& state,
                                             SessionStatistics& statistics)
{
    if (!model.expert_cache
        || model.opt.num_concurrent_sessions != 1
        || router_input.rows() != 1
        || layer.layer_id >= state.layers.size()
        || layer.moe.router_weight == invalid_tensor_handle
        || layer.moe.token_experts != invalid_tensor_handle
        || !has_flag(model.opt.flags, OptionRouterPrediction))
    {
        return nullptr;
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
        return nullptr;

    LayerCache& source_cache = state.layers[layer.layer_id];
    set_prefetch(source_cache,
                 next_layer->moe.top_k);
    adapt_prefetch(source_cache,
                   statistics,
                   has_flag(model.opt.flags, OptionRankAdaptivePrefetch));
    return next_layer;
}

static Result<RouterPredictionOutcome> run_predict(const CompiledModel& model,
                                                   const CompiledLayerPlan& next_layer,
                                                   const ActivationBuffer& router_input,
                                                   uint32_t prefetch_width)
{
    ActivationBuffer predicted_logits;
    forward_linear(model.weights.at(next_layer.moe.router_weight),
                   router_input,
                   predicted_logits,
                   model.opt.optimization_flags,
                   model.operators.find_weight(next_layer.moe.router_weight));
    if (next_layer.moe.router_bias != invalid_tensor_handle)
    {
        add_bias_inplace(predicted_logits,
                         model.weights.at(next_layer.moe.router_bias));
    }

    ExpertDispatchOptions options;
    options.expert_count = static_cast<uint32_t>(next_layer.moe.experts.size());
    options.top_k = next_layer.moe.top_k;
    options.score_function = next_layer.moe.score_function;
    options.normalization = next_layer.moe.normalization;
    options.routed_scaling_factor = next_layer.moe.routed_scaling_factor;
    options.jitter = next_layer.moe.jitter;
    if (next_layer.moe.router_selection_bias != invalid_tensor_handle)
    {
        options.selection_bias = model.weights.at(next_layer.moe.router_selection_bias).float32_values();
    }
    ExpertDispatchPlan predicted_plan;
    auto dispatched = forward_router(predicted_logits.values(),
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
        rank_experts(predicted_plan, ranked);
        predicted_experts = ranked;
    }
    else
    {
        predicted_fallback.resize(next_layer.moe.top_k);
        rank_experts(predicted_plan, predicted_fallback);
        predicted_experts = predicted_fallback;
    }
    RouterPredictionOutcome outcome;
    outcome.predicted_expert_ids.assign(next_layer.moe.top_k,
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
        if (predicted.cache_key.empty())
            continue;
        const auto prediction = model.expert_cache->prefetch_pair(gate_up,
                                                                  down,
                                                                  next_layer.layer_id,
                                                                  predicted.cache_key);
        if (prediction && prediction.value())
            ++outcome.cache_hits;
        else
            ++outcome.cache_misses;
    }
    return outcome;
}

static Result<void> apply_predict(uint32_t target_layer_id,
                                  RouterPredictionOutcome outcome,
                                  SessionState& state,
                                  SessionStatistics& statistics)
{
    if (target_layer_id >= state.layers.size())
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
        ++statistics.expert_route_rank_predictions[rank];
    }
    state.layers[target_layer_id].predicted_expert_ids = std::move(outcome.predicted_expert_ids);
    return {};
}

static void admit_predict(const CompiledModel& model,
                          const CompiledLayerPlan& layer,
                          SessionState& state,
                          size_t token_count)
{
    if (!model.expert_cache
        || !model.expert_backend
        || layer.layer_id >= state.layers.size())
    {
        return;
    }

    LayerCache& cache = state.layers[layer.layer_id];
    if (cache.predicted_expert_ids.empty())
        return;

    std::array<uint32_t, maximum_expert_route_ranks> admitted_ids{};
    size_t admitted_count = 0;
    for (const uint32_t expert_id : cache.predicted_expert_ids)
    {
        if (expert_id == std::numeric_limits<uint32_t>::max()
            || expert_id >= layer.moe.experts.size()
            || admitted_count == admitted_ids.size()
            || std::find(admitted_ids.begin(),
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
            || !support_vulkan_expert(expert, gate_up, down, model.opt.optimization_flags))
        {
            continue;
        }

        const ExpertCachePairRequest request{
            &gate_up,
            &down,
            layer.layer_id,
            expert.cache_key,
            victim_metadata(model, expert, 1)};
        ExpertCacheLease lease;
        auto ready = model.expert_cache->try_acquire_ready_pairs(std::span<const ExpertCachePairRequest>(&request, 1),
                                                                 std::span<ExpertCacheLease>(&lease, 1));
        if (!ready || !ready.value())
            continue;

        if (token_count >= vulkan_expert_gpu_admission_min_rows
            || model.opt.hybrid_mode == HybridMode::HybridExperts)
        {
            ExpertDemandRequest demand;
            demand.key = expert.cache_key;
            demand.gate_up = std::move(lease.gate_up);
            demand.gate_up_bias = expert.gate_up_bias == invalid_tensor_handle
                                      ? nullptr
                                      : &model.weights.at(expert.gate_up_bias);
            demand.down = std::move(lease.down);
            demand.down_bias = expert.down_bias == invalid_tensor_handle
                                   ? nullptr
                                   : &model.weights.at(expert.down_bias);
            demand.residency_group = layer.layer_id;
            demand.activation_limit = expert.activation_limit;
            demand.activation = expert.activation;
            model.expert_backend->admit(demand);
        }
        admitted_ids[admitted_count++] = expert_id;
    }
}

static Result<void> finish_predict(const CompiledModel& model,
                                   uint32_t layer_id,
                                   PendingRouterPrediction& pending,
                                   SessionState& state,
                                   SessionStatistics& statistics,
                                   size_t token_count)
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
    const uint32_t target_layer_id = pending.target_layer_id;
    pending.target_layer_id = std::numeric_limits<uint32_t>::max();
    if (!completed)
        return completed.error();
    auto applied = apply_predict(target_layer_id,
                                 std::move(completed).value(),
                                 state,
                                 statistics);
    if (!applied)
        return applied.error();

    if (layer_id < model.graph.layer_plans.size())
    {
        // Admit predicted weights to the GPU queue after the host pair is ready.
        admit_predict(model,
                      model.graph.layer_plans[layer_id],
                      state,
                      token_count);
    }
    return {};
}

static Result<void> prefetch_routes(const CompiledModel& model,
                                    const CompiledLayerPlan& layer,
                                    const ActivationBuffer& router_input,
                                    SessionState& state,
                                    SessionStatistics& statistics,
                                    PendingRouterPrediction& pending)
{
    const CompiledLayerPlan* next_layer = next_predict(model,
                                                       layer,
                                                       router_input,
                                                       state,
                                                       statistics);
    if (!next_layer)
        return {};

    const LayerCache& source_cache = state.layers[layer.layer_id];
    const uint32_t prefetch_width = std::min(source_cache.router_prefetch_width,
                                             next_layer->moe.top_k);

    if (has_flag(model.opt.flags, OptionAsyncRouterPrediction))
    {
        try
        {
            if (!state.router_prediction_worker)
            {
                state.router_prediction_worker = std::make_unique<CpuTaskWorker>(2);
            }
            ActivationBuffer copied_input = router_input;
            auto promise = std::make_shared<std::promise<Result<RouterPredictionOutcome>>>();
            std::future<Result<RouterPredictionOutcome>> future = promise->get_future();
            if (state.router_prediction_worker->try_submit([&model,
                                                            next_layer,
                                                            copied_input = std::move(copied_input),
                                                            prefetch_width,
                                                            promise]() mutable {
                    try
                    {
                        promise->set_value(run_predict(model,
                                                       *next_layer,
                                                       copied_input,
                                                       prefetch_width));
                    }
                    catch (const std::exception& error)
                    {
                        promise->set_value(Error{ErrorCode::InternalError,
                                                 std::string("Router prediction failed: ")
                                                     + error.what()});
                    }
                    catch (...)
                    {
                        promise->set_value(Error{ErrorCode::InternalError, "Router prediction failed"});
                    }
                }))
            {
                pending.target_layer_id = next_layer->layer_id;
                pending.result = std::move(future);
                return {};
            }
        }
        catch (...)
        {
            // Fall back to synchronous prediction.
        }
    }

    auto outcome = run_predict(model,
                               *next_layer,
                               router_input,
                               prefetch_width);
    if (!outcome)
        return outcome.error();

    auto applied = apply_predict(next_layer->layer_id,
                                 std::move(outcome).value(),
                                 state, statistics);
    if (!applied)
        return applied.error();

    admit_predict(model, *next_layer, state, router_input.rows());
    return {};
}

} // namespace moe
} // namespace ncnn
