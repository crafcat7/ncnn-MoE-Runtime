#include "gpusegment_vulkan.h"
#include "attention_vulkan.h"
#include "expertbank_vulkan.h"
#include "gateddeltanet_vulkan.h"
#include "latentattention_vulkan.h"
#include "qsaattention_vulkan.h"
#include "layernorm_vulkan.h"
#include "linear.h"
#include "modelpipeline.h"
#include "rmsnorm_vulkan.h"
#include "router_vulkan.h"
#include "sharedffn_vulkan.h"
#include "vulkancontext.h"
#include "graph/compiledmodel.h"
#include "graph/memoryplan.h"
#include "graph/submissionbudget.h"
#include "kernels/statecache.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <layer.h>
#include <layer_type.h>
#include <paramdict.h>
#endif

namespace ncnn {
namespace moe {

class GpuSegmentWorkspace_vulkan::Implementation
{
public:
    std::shared_ptr<const DeviceTensor_vulkan> normalized;
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanHostOutput> host_output;
    uint64_t host_output_slot_size = 0;
    std::shared_ptr<VulkanContext> context;
    ncnn::Option option;
    std::shared_ptr<ncnn::Layer> add;
    std::shared_ptr<SharedFfn_vulkan> shared;
    std::shared_ptr<const DeviceTensor_vulkan> residual;
    std::shared_ptr<const DeviceTensor_vulkan> combined;
    bool dense_ready = false;

    bool record_hidden(ncnn::VkMat& output, ncnn::VkCompute& command,
                       std::vector<ncnn::VkMat>& storage, FfnRecordStatistics_vulkan& statistics,
                       const ncnn::VkMat* recorded_combined = nullptr) const
    {
        const auto* aggregate = recorded_combined ? recorded_combined : combined ? combined->value()
                                                                                 : nullptr;
        if (!context || !add || !normalized || !residual || !aggregate
            || normalized->empty() || residual->empty() || aggregate->empty()
            || !normalized->value() || !residual->value() || aggregate->dims != 2
            || aggregate->elempack != 1 || aggregate->elemsize != sizeof(float)
            || !aggregate->allocator || aggregate->allocator->vkdev != context->device()
            || residual->rows() != static_cast<size_t>(aggregate->h)
            || residual->columns() != static_cast<uint32_t>(aggregate->w)) return false;
        residual->prepare_read();
        // A just-recorded aggregate already carries its producer's access
        // state. Only completed operands need cross-command rearming.
        if (!recorded_combined) combined->prepare_read();
        ncnn::VkMat sum = *aggregate;
        if (shared)
        {
            normalized->prepare_read();
            ncnn::VkMat shared_output;
            if (!shared->record(*normalized->value(), shared_output, command, storage, statistics)) return false;
            storage.push_back(shared_output);
            std::vector<ncnn::VkMat> added(1);
            if (add->forward({sum, shared_output}, added, command, option) != 0 || added[0].empty()) return false;
            sum = added[0];
            storage.push_back(sum);
            ++statistics.dispatches;
        }
        std::vector<ncnn::VkMat> added(1);
        if (add->forward({*residual->value(), sum}, added, command, option) != 0 || added[0].empty()) return false;
        output = added[0];
        storage.push_back(output);
        ++statistics.dispatches;
        return true;
    }
#endif
};

class GpuSegment_vulkan::Implementation
{
public:
    Implementation();
    ~Implementation();
    std::shared_ptr<VulkanContext> context;
#if NCNN_MOE_WITH_VULKAN
    class ResidentWorkspace;
    std::unique_ptr<ResidentWorkspace> resident_workspace;
    ncnn::Option option;
    std::shared_ptr<Attention_vulkan> attention;
    std::shared_ptr<GatedDeltaNet_vulkan> delta;
    std::shared_ptr<LatentAttention_vulkan> mla;
    std::shared_ptr<QsaAttention_vulkan> qsa;
    std::shared_ptr<RmsNorm_vulkan> attention_norm;
    std::shared_ptr<RmsNorm_vulkan> ffn_norm;
    std::shared_ptr<LayerNorm_vulkan> layer_norm;
    std::shared_ptr<Router_vulkan> router;
    std::shared_ptr<SharedFfn_vulkan> shared;
    std::shared_ptr<SharedFfn_vulkan> dense;
    std::shared_ptr<ncnn::Layer> add;
    CompiledOperator lm_head;
    uint32_t hidden = 0;
    bool add_norm = false;
    bool head = false;
#endif
};

GpuSegmentWorkspace_vulkan::GpuSegmentWorkspace_vulkan() : d(new Implementation)
{
}
GpuSegmentWorkspace_vulkan::~GpuSegmentWorkspace_vulkan() = default;
void GpuSegmentWorkspace_vulkan::reset()
{
#if NCNN_MOE_WITH_VULKAN
    Implementation next;
    next.host_output = std::move(d->host_output);
    next.host_output_slot_size = d->host_output_slot_size;
    *d = std::move(next);
#else
    *d = Implementation{};
#endif
}
void GpuSegmentWorkspace_vulkan::reserve_host_output(uint64_t slot_capacity) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    d->host_output_slot_size = std::max(d->host_output_slot_size, slot_capacity);
#else
    (void)slot_capacity;
#endif
}
bool GpuSegmentWorkspace_vulkan::active() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->context && d->normalized && d->residual;
#else
    return false;
#endif
}
uint64_t GpuSegmentWorkspace_vulkan::allocated_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    uint64_t bytes = d->host_output ? d->host_output->allocated_bytes() : 0;
    for (const auto& value : {d->normalized, d->residual, d->combined})
        if (value && !value->empty()) bytes += value->value()->total() * value->value()->elemsize;
    return bytes;
#else
    return 0;
#endif
}
const std::shared_ptr<const DeviceTensor_vulkan>& GpuSegmentWorkspace_vulkan::device_input() const noexcept
{ return d->normalized; }
GpuSegment_vulkan::GpuSegment_vulkan() : d(new Implementation)
{
}
GpuSegment_vulkan::~GpuSegment_vulkan() = default;
const std::shared_ptr<VulkanContext>& GpuSegment_vulkan::context() const noexcept
{ return d->context; }

#if NCNN_MOE_WITH_VULKAN
static std::shared_ptr<ncnn::Layer> segment_add(const std::shared_ptr<VulkanContext>& context, const ncnn::Option& option)
{
    auto result = std::shared_ptr<ncnn::Layer>(ncnn::create_layer_vulkan(ncnn::LayerType::BinaryOp), [context, option](ncnn::Layer* layer) {
        if (!layer) return;
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        layer->destroy_pipeline(option);
        delete layer;
    });
    if (!result) return {};
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    result->vkdev = context->device();
    ncnn::ParamDict parameters;
    parameters.set(0, 0);
    if (result->load_param(parameters) != 0 || result->create_pipeline(option) != 0) return {};
    return result;
}

static bool segment_float_vector(const CompiledModel& model, TensorHandle handle, uint32_t columns) noexcept
{
    if (handle >= model.weights.size()) return false;
    const auto& value = model.weights.at(handle);
    return value.shape.size() == 1 && value.shape[0] == columns
           && ((value.dtype == DType::Float32 && value.float32_values().size() == columns)
               || (value.dtype == DType::BFloat16 && value.bfloat16_values().size() == columns));
}

static bool segment_norm_supported(const CompiledModel& model, TensorHandle weight, TensorHandle bias, uint32_t columns) noexcept
{
    return segment_float_vector(model, weight, columns)
           && (bias == invalid_tensor_handle || segment_float_vector(model, bias, columns));
}

static bool reset_segment_command(VulkanTransferSlot& slot)
{
    if (slot.command_used && slot.command->reset() != 0) return false;
    slot.command_used = true;
    return true;
}
#endif

bool GpuSegment_vulkan::supports(const CompiledModel& model, const CompiledLayerPlan& layer) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    if (model.opt.hybrid_mode != HybridMode::HybridExperts || has_flag(model.opt.flags, OptionRouterPrediction)
        || !has_flag(model.opt.optimization_flags, OptimizationVulkanCommandGraph)
        || model.opt.vulkan_device_indices.size() > 1 || model.descriptor.hyper_connection_kind != HyperConnectionKind::None
        || model.descriptor.hyper_connection_multiplier != 1 || layer.ple.enabled()) return false;
    const bool dense = layer.dense_ffn.down_weight != invalid_tensor_handle;
    if (dense)
    {
        const auto& operation = model.operators.at(layer.dense_ffn.vulkan_operator);
        const bool dense_ffn = operation.qnk_expert
                                   ? operation.qnk_expert->vulkan_context() != nullptr
                                   : operation.bfloat16
                                         && model.operators.at_weight(layer.dense_ffn.down_weight).bfloat16;
        if (layer.dense_ffn.use_gpu_cache || !dense_ffn
            || model.speculative.enabled()
            || !segment_norm_supported(model, layer.dense_pre_ffn_norm_weight, layer.dense_pre_ffn_norm_bias,
                                       model.descriptor.hidden_size)) return false;
    }
    else
    {
        if (layer.moe.experts.empty() || layer.moe.top_k == 0 || layer.moe.top_k > 64 || layer.moe.experts.size() > 2048
            || !segment_norm_supported(model, layer.moe.pre_ffn_norm_weight, layer.moe.pre_ffn_norm_bias,
                                       model.descriptor.hidden_size)
            || layer.moe.router_weight >= model.weights.size()) return false;
        const auto& router = model.weights.at(layer.moe.router_weight);
        if (router.shape.size() != 2 || router.shape[0] != layer.moe.experts.size()
            || router.shape[1] != model.descriptor.hidden_size
            || (router.dtype != DType::Float32 && router.dtype != DType::BFloat16)) return false;
        if (layer.moe.has_shared_expert && !support_vulkan_shared_experts(model.operators, layer.moe)) return false;
    }
    if (layer.attention.kind == AttentionKind::GatedDeltaNet)
    {
        const auto& operation = model.operators.at(layer.attention.gated_delta_vulkan_operator).gated_delta;
        return operation && activation_variant(operation->option()) == 0;
    }
    if (layer.attention.kind == AttentionKind::MultiHeadLatent)
    {
        if (!has_flag(layer.attention.flags, AttentionBlockLatentValue))
            return false;
        const auto& operation = model.operators.at_weight(layer.attention.query_a_weight).latent_attention;
        const bool supported = operation && operation->can_record()
                               && activation_variant(operation->option()) == 0;
        return supported;
    }
    if (layer.attention.kind != AttentionKind::Standard) return false;
    const auto& operation = model.operators.at(layer.attention.vulkan_attention_operator);
    return has_flag(layer.attention.flags, AttentionBlockQsa)
               ? operation.qsa_attention && activation_variant(operation.qsa_attention->option()) == 0
               : operation.attention && activation_variant(operation.attention->option()) == 0;
#else
    (void)model;
    (void)layer;
    return false;
#endif
}

bool GpuSegment_vulkan::supports_head(const CompiledModel& model) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    if (model.opt.hybrid_mode != HybridMode::HybridExperts || !has_flag(model.opt.optimization_flags, OptimizationVulkanCommandGraph)
        || model.descriptor.hyper_connection_kind != HyperConnectionKind::None
        || model.descriptor.hyper_connection_multiplier != 1
        || !segment_norm_supported(model, model.final_norm_weight, model.final_norm_bias,
                                   model.descriptor.hidden_size)) return false;
    const auto& operation = model.operators.at_weight(model.lm_head_weight);
    if (operation.qnk) return operation.qnk->vulkan_context() && activation_variant(operation.qnk->option()) == 0;
    if (operation.float8) return operation.float8->vulkan_context() && activation_variant(operation.float8->option()) == 0;
    if (operation.bfloat16) return operation.bfloat16->vulkan_context() && activation_variant(operation.bfloat16->option()) == 0;
    return operation.linear && operation.linear->vulkan_context() && activation_variant(operation.linear->option()) == 0;
#else
    (void)model;
    return false;
#endif
}

std::shared_ptr<GpuSegment_vulkan> GpuSegment_vulkan::create(const CompiledModel& model, const CompiledLayerPlan& layer)
{
#if NCNN_MOE_WITH_VULKAN
    if (!supports(model, layer)) return {};
    auto result = std::shared_ptr<GpuSegment_vulkan>(new GpuSegment_vulkan);
    auto& impl = *result->d;
    const auto& attention = layer.attention;
    if (attention.kind == AttentionKind::GatedDeltaNet)
    {
        impl.delta = model.operators.at(attention.gated_delta_vulkan_operator).gated_delta;
        impl.context = impl.delta->vulkan_context();
        impl.option = impl.delta->option();
        if (!impl.delta->has_input_rms_norm() && attention.pre_attention_norm_weight != invalid_tensor_handle)
        {
            impl.attention_norm = RmsNorm_vulkan::create(model.weights.at(attention.pre_attention_norm_weight), attention.norm_epsilon,
                                                         attention.norm_weight_offset, impl.context, impl.option);
            if (!impl.attention_norm) return {};
        }
    }
    else if (attention.kind == AttentionKind::MultiHeadLatent)
    {
        impl.mla = model.operators.at_weight(attention.query_a_weight).latent_attention;
        impl.context = impl.mla->vulkan_context();
        impl.option = impl.mla->option();
        if (attention.pre_attention_norm_weight != invalid_tensor_handle)
        {
            impl.attention_norm = RmsNorm_vulkan::create(model.weights.at(attention.pre_attention_norm_weight),
                                                         attention.norm_epsilon, attention.norm_weight_offset,
                                                         impl.context, impl.option);
            if (!impl.attention_norm) return {};
        }
    }
    else if (has_flag(attention.flags, AttentionBlockQsa))
    {
        impl.qsa = model.operators.at(attention.vulkan_attention_operator).qsa_attention;
        impl.context = impl.qsa->vulkan_context();
        impl.option = impl.qsa->option();
    }
    else
    {
        impl.attention = model.operators.at(attention.vulkan_attention_operator).attention;
        impl.context = impl.attention->vulkan_context();
        impl.option = impl.attention->option();
    }
    impl.hidden = model.descriptor.hidden_size;
    const auto& limits = impl.context->device()->info.physicalDeviceProperties().limits;
    const bool dense = layer.dense_ffn.down_weight != invalid_tensor_handle;
    const TensorHandle ffn_norm_weight = dense ? layer.dense_pre_ffn_norm_weight : layer.moe.pre_ffn_norm_weight;
    const TensorHandle ffn_norm_bias = dense ? layer.dense_pre_ffn_norm_bias : layer.moe.pre_ffn_norm_bias;
    const bool use_layer_norm = ffn_norm_bias != invalid_tensor_handle;
    impl.add_norm = !use_layer_norm
                    && limits.maxPerStageDescriptorStorageBuffers >= 5
                    && limits.maxDescriptorSetStorageBuffers >= 5;
    if (use_layer_norm)
    {
        impl.layer_norm = LayerNorm_vulkan::create(model.weights.at(ffn_norm_weight),
                                                   model.weights.at(ffn_norm_bias),
                                                   model.descriptor.norm_epsilon,
                                                   impl.context, impl.option);
    }
    else
    {
        impl.ffn_norm = RmsNorm_vulkan::create(model.weights.at(ffn_norm_weight), model.descriptor.norm_epsilon,
                                               model.descriptor.norm_weight_offset, impl.context, impl.option,
                                               nullptr, impl.add_norm);
    }
    if (dense)
        impl.dense = SharedFfn_vulkan::create(model, layer.dense_ffn);
    else
    {
        impl.router = Router_vulkan::create(model.weights.at(layer.moe.router_weight),
                                            layer.moe.router_bias == invalid_tensor_handle ? nullptr : &model.weights.at(layer.moe.router_bias),
                                            layer.moe.router_selection_bias == invalid_tensor_handle ? nullptr : &model.weights.at(layer.moe.router_selection_bias),
                                            layer.vulkan_device_index, model.vulkan_runtime, model.opt.optimization_flags);
        if (layer.moe.has_shared_expert) impl.shared = SharedFfn_vulkan::create(model, layer.moe);
    }
    impl.add = segment_add(impl.context, impl.option);
    if ((!impl.ffn_norm && !impl.layer_norm) || !impl.add
        || (dense ? !impl.dense || impl.dense->context() != impl.context
                  : !impl.router || impl.router->context() != impl.context)
        || (layer.moe.has_shared_expert && (!impl.shared || impl.shared->context() != impl.context))) return {};
    return result;
#else
    (void)model;
    (void)layer;
    return {};
#endif
}

std::shared_ptr<GpuSegment_vulkan> GpuSegment_vulkan::create_head(const CompiledModel& model)
{
#if NCNN_MOE_WITH_VULKAN
    if (!supports_head(model)) return {};
    auto result = std::shared_ptr<GpuSegment_vulkan>(new GpuSegment_vulkan);
    auto& impl = *result->d;
    impl.head = true;
    impl.hidden = model.descriptor.hidden_size;
    impl.lm_head = model.operators.at_weight(model.lm_head_weight);
    if (impl.lm_head.qnk)
    {
        impl.context = impl.lm_head.qnk->vulkan_context();
        impl.option = impl.lm_head.qnk->option();
    }
    else if (impl.lm_head.float8)
    {
        impl.context = impl.lm_head.float8->vulkan_context();
        impl.option = impl.lm_head.float8->option();
    }
    else if (impl.lm_head.bfloat16)
    {
        impl.context = impl.lm_head.bfloat16->vulkan_context();
        impl.option = impl.lm_head.bfloat16->option();
    }
    else
    {
        impl.context = impl.lm_head.linear->vulkan_context();
        impl.option = impl.lm_head.linear->option();
    }
    impl.option.use_packing_layout = false;
    if (model.final_norm_bias != invalid_tensor_handle)
    {
        impl.layer_norm = LayerNorm_vulkan::create(model.weights.at(model.final_norm_weight),
                                                   model.weights.at(model.final_norm_bias),
                                                   model.descriptor.norm_epsilon,
                                                   impl.context, impl.option);
    }
    else
    {
        impl.ffn_norm = RmsNorm_vulkan::create(model.weights.at(model.final_norm_weight), model.descriptor.norm_epsilon,
                                               model.descriptor.norm_weight_offset, impl.context, impl.option);
    }
    impl.add = segment_add(impl.context, impl.option);
    if ((!impl.ffn_norm && !impl.layer_norm) || !impl.add) return {};
    return result;
#else
    (void)model;
    return {};
#endif
}

Result<uint32_t> GpuSegment_vulkan::continuation_target(const CompiledModel& model, uint32_t expert_node_id,
                                                        uint32_t& fence_id)
{
    fence_id = invalid_layer_id;
    if (expert_node_id >= model.net.layers.size()) return invalid_layer_id;
    const auto& source = model.net.layers[expert_node_id];
    if (source.kind != LayerKind::ExpertGroup || source.backend != ExecutionBackend::Vulkan
        || source.plan_index >= model.net.layer_plans.size()
        || source.plan_index >= model.gpu_segments.size() || !model.gpu_segments[source.plan_index])
        return invalid_layer_id;
    const auto& source_layer = model.net.layer_plans[source.plan_index];
    const auto& context = model.gpu_segments[source.plan_index]->context();
    if (!context) return invalid_layer_id;
    auto next = std::find(model.net.order.begin(), model.net.order.end(), expert_node_id);
    if (next == model.net.order.end()) return invalid_layer_id;
    bool combined = false;
    uint32_t target_id = invalid_layer_id;
    bool head = false;
    while (++next != model.net.order.end())
    {
        const LayerId candidate_id = *next;
        if (candidate_id >= model.net.layers.size()) return invalid_layer_id;
        const auto& candidate = model.net.layers[candidate_id];
        if (!combined && candidate.plan_index == source.plan_index
            && candidate.backend == ExecutionBackend::Vulkan && candidate.kind == LayerKind::SharedExpertGroup)
            continue;
        if (!combined && candidate.plan_index == source.plan_index
            && candidate.backend == ExecutionBackend::Vulkan && candidate.kind == LayerKind::Combine)
        {
            combined = true;
            continue;
        }
        if (!combined || candidate.backend != ExecutionBackend::Vulkan) return invalid_layer_id;
        if (candidate.kind == LayerKind::Attention
            && candidate.plan_index < model.net.layer_plans.size()
            && candidate.plan_index < model.gpu_segments.size()
            && candidate.plan_index != source.plan_index
            && model.net.layer_plans[candidate.plan_index].vulkan_device_index == source_layer.vulkan_device_index
            && model.gpu_segments[candidate.plan_index]
            && model.gpu_segments[candidate.plan_index]->context() == context)
        {
            target_id = candidate_id;
            break;
        }
        if (candidate.kind == LayerKind::FinalNorm && model.gpu_head && model.gpu_head->context() == context)
        {
            target_id = candidate_id;
            head = true;
            break;
        }
        // All other nodes are explicit boundaries. Follow actual schedule
        // order rather than assuming that layer ids are consecutive.
        return invalid_layer_id;
    }
    if (target_id == invalid_layer_id) return invalid_layer_id;

    uint32_t resolved_fence = invalid_layer_id;
    auto fence = next;
    while (++fence != model.net.order.end())
    {
        const LayerId candidate_id = *fence;
        if (candidate_id >= model.net.layers.size())
            return Error{ErrorCode::InvalidModel, "GPU continuation memory target has an invalid scheduled fence"};
        const auto& candidate = model.net.layers[candidate_id];
        if ((head && candidate.kind == LayerKind::LmHead)
            || (!head && candidate.kind == LayerKind::Router
                && candidate.plan_index == model.net.layers[target_id].plan_index))
        {
            resolved_fence = candidate_id;
            break;
        }
        // Match the memory planner's existing stop points. In particular, do
        // not search beyond another layer's Expert/Attention/FFN boundary.
        if (candidate.kind == LayerKind::ExpertGroup || candidate.kind == LayerKind::Attention
            || candidate.kind == LayerKind::DenseFfn || candidate.kind == LayerKind::TokenEmbedding)
            break;
    }
    if (resolved_fence == invalid_layer_id || model.net.layers[resolved_fence].backend != ExecutionBackend::Vulkan)
        return Error{ErrorCode::InvalidModel, "GPU continuation memory target has no selected Router or LM head fence"};

    const auto device_index = [&](const Layer& node) {
        return node.plan_index < model.net.layer_plans.size()
                   ? model.net.layer_plans[node.plan_index].vulkan_device_index
                   : model.opt.vulkan_device_index;
    };
    const uint32_t device = device_index(source);
    if (device_index(model.net.layers[target_id]) != device || device_index(model.net.layers[resolved_fence]) != device)
        return Error{ErrorCode::InvalidModel, "GPU continuation memory submission crosses a device boundary"};

    fence_id = resolved_fence;
    return target_id;
}

class GpuSegmentContinuation_vulkan::Implementation
{
public:
    std::shared_ptr<GpuSegment_vulkan::Implementation> target;
    std::string error;
    bool recorded = false, attempted = false, completed = false, published = false, failed = false;
#if NCNN_MOE_WITH_VULKAN
    GpuSegmentWorkspace_vulkan::Implementation previous;
    std::vector<uint64_t> positions;
    std::vector<LayerCache*> caches;
    ExpertDispatchOptions options;
    std::vector<float> selection_bias;
    std::vector<uint32_t> explicit_ids;
    const ActivationBuffer* initial_hidden = nullptr;
    std::vector<ncnn::VkMat> storage;
    std::vector<std::unique_ptr<AttentionWork_vulkan>> attention_work;
    size_t attention_work_used = 0;
    std::vector<std::unique_ptr<GatedDeltaWork_vulkan>> delta_work;
    std::vector<std::unique_ptr<LatentValueWork_vulkan>> mla_work;
    std::vector<std::unique_ptr<QsaAttentionWork_vulkan>> qsa_work;
    RouterWorkspace_vulkan router_work;
    FfnRecordStatistics_vulkan ffn_statistics;
    std::shared_ptr<VulkanHostOutput> host_output;
    uint64_t host_output_slot_size = 0;
    ActivationBuffer pending_logits;
    // Mapped views are destroyed before their lease and dedicated allocator.
    ncnn::VkMat upload, input, normalized_attention, branch, residual, normalized, output, selected_staging;
    ncnn::VkMat dense_output;
    uint32_t selected_columns = 0;
    size_t output_rows = 0;
    bool uploaded = false, standalone = false, produce_logits = false, last_row_only = false, fused_head = false;

    std::span<std::unique_ptr<AttentionWork_vulkan>> active_attention_work() noexcept
    {
        return {attention_work.data(), attention_work_used};
    }

    // Called only after command retirement and publication/failure handling.
    // Keep empty metadata capacity; no allocation or Session owner is deferred.
    void retire() noexcept
    {
        for (auto& work : active_attention_work()) work->retire();
        attention_work_used = 0;
        // These tickets do not yet expose a reusable-state API.
        delta_work.clear();
        mla_work.clear();
        qsa_work.clear();
        storage.clear();
        router_work = {};
        for (auto* value : {&upload, &input, &normalized_attention, &branch,
                            &residual, &normalized, &output, &selected_staging, &dense_output})
            value->release();
        pending_logits.clear();
        host_output.reset();
        host_output_slot_size = 0;
        previous = {};
        positions.clear();
        caches.clear();
        selection_bias.clear();
        explicit_ids.clear();
        options = {};
        initial_hidden = nullptr;
        ffn_statistics = {};
        selected_columns = 0;
        output_rows = 0;
        uploaded = standalone = produce_logits = last_row_only = fused_head = false;
        recorded = attempted = completed = published = failed = false;
        error.clear();
        // In particular, a head-owned pool must not retain its own target.
        target.reset();
    }

    bool record_layer(const ncnn::VkMat* combined, ncnn::VkCompute& command,
                      const ncnn::VkMat* recorded_hidden = nullptr, bool download_routes = true)
    {
        const auto* d = target.get();
        auto fail = [&](const char* reason) { error = reason; return false; };
        uploaded = !previous.context && !recorded_hidden;
        if (recorded_hidden)
            input = *recorded_hidden;
        else if (previous.context)
        {
            if (!previous.record_hidden(input, command, storage, ffn_statistics, combined)) return fail("cannot record prior Vulkan FFN residual");
        }
        else if (!fill_staging_upload(*initial_hidden, upload, d->context->staging_allocator())
                 || !record_mapped_upload(upload, input, command, d->option))
            return fail("cannot upload initial Vulkan hidden");
        if (d->attention_norm)
        {
            if (!d->attention_norm->record(input, normalized_attention, command)) return fail("cannot record pre-attention RMSNorm");
            ++ffn_statistics.dispatches;
        }
        else
            normalized_attention = input;
        const bool single_cache = std::all_of(caches.begin(), caches.end(), [&](LayerCache* cache) { return cache == caches.front(); });
        if (!single_cache)
        {
            branch.create(d->hidden, static_cast<int>(positions.size()), sizeof(float), d->context->blob_allocator());
            if (branch.empty()) return fail("cannot allocate Vulkan Attention branch");
        }
        // Group adjacent rows from one cache. Prefill retains native multi-row
        // kernels; independent Session rows are all recorded into this command.
        for (size_t first = 0; first < positions.size();)
        {
            size_t count = 1;
            while (first + count < positions.size() && caches[first + count] == caches[first]
                   && positions[first + count] == positions[first] + count) ++count;
            for (size_t prior = 0; prior < first; ++prior)
                if (caches[prior] == caches[first]) return fail("interleaved rows of one cache require contiguous Vulkan recording");
            ncnn::VkMat source = normalized_attention;
            if (first != 0 || count != positions.size())
            {
                // Clone the view so every storage binding meets the device's
                // offset alignment, including small synthetic hidden widths.
                const ncnn::VkMat view = row_view(normalized_attention, first, count);
                if (view.empty()) return fail("invalid Vulkan Session row view");
                source = view;
                if (source.buffer_offset() % std::max<size_t>(4, d->context->device()->info.buffer_offset_alignment()) != 0)
                {
                    ncnn::VkMat aligned;
                    command.record_clone(view, aligned, d->option);
                    if (aligned.empty()) return fail("cannot allocate aligned Vulkan Session row view");
                    storage.push_back(aligned);
                    source = aligned;
                }
            }
            ncnn::VkMat output;
            if (d->attention)
            {
                if (attention_work_used == attention_work.size())
                    attention_work.push_back(std::make_unique<AttentionWork_vulkan>());
                auto& work = *attention_work[attention_work_used++];
                if (!d->attention->record_forward(positions[first], *caches[first], source, output, command, work, false)) return fail("Standard Attention device recording failed");
            }
            else if (d->delta)
            {
                delta_work.push_back(std::make_unique<GatedDeltaWork_vulkan>());
                if (positions[first] != caches[first]->gated_delta_token_count
                    || !d->delta->record_forward(source, *caches[first], count, output, command, *delta_work.back(), d->delta->has_input_rms_norm())) return fail("GatedDeltaNet device recording failed");
            }
            else if (d->mla)
            {
                mla_work.push_back(std::make_unique<LatentValueWork_vulkan>());
                if (!d->mla->record_value_forward(source,
                                                  std::span<const uint64_t>(positions).subspan(first, count),
                                                  std::span<LayerCache* const>(caches).subspan(first, count),
                                                  output, command, *mla_work.back(), storage))
                    return fail("expanded-value latent attention device recording failed");
            }
            else
            {
                qsa_work.push_back(std::make_unique<QsaAttentionWork_vulkan>());
                if (!d->qsa->record_forward(source, *caches[first], positions[first], output, command, *qsa_work.back())) return fail("QSA device recording failed");
            }
            if (first == 0 && count == positions.size())
                branch = output;
            else
            {
                storage.push_back(output);
                ncnn::VkMat destination = row_view(branch, first, count);
                const auto buffer = destination.buffer();
                const auto offset = destination.buffer_offset();
                command.record_clone(row_view(output, 0, count), destination, d->option);
                if (destination.empty() || destination.buffer() != buffer || destination.buffer_offset() != offset)
                    return fail("cannot copy Vulkan Attention Session output into its planned row range");
            }
            first += count;
        }
        if (d->add_norm)
        {
            if (!d->ffn_norm->record(input, normalized, command, &branch, &residual))
                return fail("cannot record fused Vulkan Attention residual and FFN norm");
        }
        else
        {
            std::vector<ncnn::VkMat> added(1);
            if (d->add->forward({input, branch}, added, command, d->option) != 0 || added[0].empty()) return fail("cannot record Attention residual");
            residual = added[0];
            ++ffn_statistics.dispatches;
            const bool norm_recorded = d->layer_norm
                                           ? d->layer_norm->record(residual, normalized, command)
                                           : d->ffn_norm->record(residual, normalized, command);
            if (!norm_recorded)
                return fail("cannot record Vulkan FFN norm");
        }
        if (d->dense)
        {
            ncnn::VkMat ffn_output;
            if (!d->dense->record(normalized, ffn_output, command, storage, ffn_statistics))
                return fail("cannot record Vulkan dense FFN");
            std::vector<ncnn::VkMat> added(1);
            if (d->add->forward({residual, ffn_output}, added, command, d->option) != 0 || added[0].empty())
                return fail("cannot record Vulkan dense FFN residual");
            storage.push_back(ffn_output);
            storage.push_back(added[0]);
            ++ffn_statistics.dispatches;
            dense_output = added[0];
            return true;
        }
        if (!d->router->record(normalized, options, router_work, command))
            return fail("cannot record Vulkan router");
        selected_columns = Router_vulkan::selected_columns(options.top_k);
        if (!download_routes) return true;
        if (!prepare_staging_batch(selected_staging, positions.size(), selected_columns, d->context->staging_allocator())
            || !record_staging_download(router_work.selected, positions.size(), selected_columns,
                                        selected_staging, command, d->context->device(), d->option)) return fail("cannot record compact Vulkan routes download");

        return true;
    }

    bool record_head(const ncnn::VkMat* combined, ncnn::VkCompute& command,
                     const ncnn::VkMat* recorded_hidden = nullptr)
    {
        const auto* d = target.get();
        if (recorded_hidden)
            input = *recorded_hidden;
        else if (!previous.record_hidden(input, command, storage, ffn_statistics, combined))
        {
            error = "cannot record final Vulkan FFN residual";
            return false;
        }
        output_rows = static_cast<size_t>(input.h);
        if (produce_logits)
        {
            if (last_row_only && output_rows > 1)
            {
                const ncnn::VkMat view = row_view(input, output_rows - 1, 1);
                if (view.empty())
                {
                    error = "invalid final Vulkan hidden row";
                    return false;
                }
                input = view;
                if (view.buffer_offset() % std::max<size_t>(4, d->context->device()->info.buffer_offset_alignment()) != 0)
                {
                    ncnn::VkMat aligned;
                    command.record_clone(view, aligned, d->option);
                    if (aligned.empty())
                    {
                        error = "cannot align final Vulkan hidden row";
                        return false;
                    }
                    storage.push_back(aligned);
                    input = aligned;
                }
                output_rows = 1;
            }
            bool recorded = false;
            fused_head = d->lm_head.bfloat16 && d->lm_head.bfloat16->has_rms_norm_chain();
            if (fused_head)
            {
                recorded = d->lm_head.bfloat16->record_rms_norm_chain(input, output, command);
            }
            else
            {
                const bool norm_recorded = d->layer_norm
                                               ? d->layer_norm->record(input, normalized, command)
                                               : d->ffn_norm->record(input, normalized, command);
                if (!norm_recorded)
                {
                    error = "cannot record final Vulkan normalization";
                    return false;
                }
                if (d->lm_head.qnk)
                    recorded = d->lm_head.qnk->record_forward(normalized, output, command, storage);
                else if (d->lm_head.float8)
                    recorded = d->lm_head.float8->record_forward(normalized, output, command, storage);
                else if (d->lm_head.bfloat16)
                    recorded = d->lm_head.bfloat16->forward(normalized, output, command, d->option) == 0;
                else if (d->lm_head.linear)
                    recorded = d->lm_head.linear->forward(normalized, output, command, d->option) == 0;
            }
            if (!recorded || output.empty())
            {
                error = "cannot record resident Vulkan LM projection";
                return false;
            }
            if (output.elempack != 1)
            {
                ncnn::VkMat unpacked;
                d->context->device()->convert_packing(output, unpacked, 1, command, d->option);
                if (unpacked.empty())
                {
                    error = "cannot unpack Vulkan logits";
                    return false;
                }
                storage.push_back(output);
                output = unpacked;
            }
            if (!host_output) host_output = std::make_shared<VulkanHostOutput>();
            // Direct compiler clients plan their requested output explicitly;
            // Runtime workspaces receive the model's admitted shape bound.
            if (host_output_slot_size == 0)
            {
                const auto planned = plan_host_output_slot_size(output_rows, output.w);
                if (!planned)
                {
                    error = planned.error().message;
                    return false;
                }
                host_output_slot_size = std::max(planned.value(), host_output->planned_capacity());
            }
            if (!host_output->prepare(d->context, output_rows, output.w, host_output_slot_size, selected_staging, pending_logits)
                || !record_staging_download(output, output_rows, output.w, selected_staging, command, d->context->device(), d->option))
            {
                error = "cannot record Vulkan logits download";
                return false;
            }
        }

        return true;
    }

    bool record_work(const ncnn::VkMat* combined, ncnn::VkCompute& command)
    {
        if (recorded || failed || !target || !target->context) return false;
        if (!(target->head ? record_head(combined, command) : record_layer(combined, command)))
        {
            failed = true;
            return false;
        }
        recorded = true;
        return true;
    }

    bool same_workspace(const GpuSegmentWorkspace_vulkan::Implementation& current) const noexcept
    {
        return current.context == previous.context && current.normalized == previous.normalized
               && current.residual == previous.residual && current.combined == previous.combined;
    }
#endif
};

#if NCNN_MOE_WITH_VULKAN
class GpuSegment_vulkan::Implementation::ResidentWorkspace
{
public:
    // One pool per compiled head, serialized by its Context command mutex.
    // Slots are bounded by the largest planned chunk plus its optional head;
    // per-slot row/ticket capacity is bounded by admitted Session row counts.
    std::vector<std::unique_ptr<GpuSegmentContinuation_vulkan::Implementation>> slots;
    std::vector<GpuSegmentContinuation_vulkan::Implementation*> work;
    std::vector<ncnn::VkMat> bank_storage;
    struct LayerWorkCoefficients
    {
        double attention_router_per_row = 0;
        double routed_expert_per_selection_row = 0;
        double fixed_ffn_per_row = 0;
    };
    struct LayerWork
    {
        double attention_router = 0;
        double ffn = 0;
    };
    std::vector<LayerWorkCoefficients> layer_work_coefficients;
    std::vector<LayerWork> layer_work;
    double head_work_per_row = 0;
    const CompiledModel* work_coefficients_model = nullptr;

    void prepare_work_coefficients(const CompiledModel& model)
    {
        if (work_coefficients_model == &model && layer_work_coefficients.size() == model.net.layer_plans.size()) return;
        // A copied compiled model may share its head but own a different graph.
        // Bind this cache to the current immutable compiled model as well.
        // Inspect weight shapes once; rows and selected-expert counts remain
        // runtime inputs. Publish the cache only after all shapes are read.
        std::vector<LayerWorkCoefficients> prepared(model.net.layer_plans.size());
        const auto projection_work = [&](TensorHandle handle) -> double {
            if (handle == invalid_tensor_handle) return 0;
            const auto& shape = model.weights.at(handle).shape;
            return shape.size() == 2 ? 2.0 * shape[0] * shape[1] : 0;
        };
        const auto expert_work = [&](const ExpertPlan& expert) {
            return projection_work(expert.gate_weight) + projection_work(expert.up_weight)
                   + projection_work(expert.gate_up_weight) + projection_work(expert.down_weight);
        };
        for (size_t layer = 0; layer < prepared.size(); ++layer)
        {
            const auto& plan = model.net.layer_plans[layer];
            const auto& a = plan.attention;
            auto& coefficients = prepared[layer];
            for (auto handle : {a.query_weight, a.key_weight, a.value_weight, a.output_weight, a.output_gate_weight,
                                a.delta_qkv_weight, a.delta_z_weight, a.delta_beta_weight, a.delta_alpha_weight,
                                a.query_a_weight, a.query_b_weight, a.key_value_weight, a.output_a_weight, a.output_b_weight,
                                a.qsa_query_key_weight, plan.moe.router_weight})
                coefficients.attention_router_per_row += projection_work(handle);
            if (plan.dense_ffn.down_weight != invalid_tensor_handle)
                coefficients.fixed_ffn_per_row = expert_work(plan.dense_ffn);
            else
            {
                // Tail layers beyond a partial bank are not part of the
                // recorded prefix and keep streaming through the normal path.
                if (layer < model.gpu_expert_banks.size() && model.gpu_expert_banks[layer])
                {
                    const auto& bank = model.gpu_expert_banks[layer]->description();
                    coefficients.routed_expert_per_selection_row = 2.0 * bank.intermediate_columns
                                                                   * (2.0 * bank.input_columns + bank.output_columns);
                }
                if (plan.moe.has_shared_expert)
                    coefficients.fixed_ffn_per_row = expert_work(plan.moe.shared_expert);
            }
        }
        const double prepared_head_work = projection_work(model.lm_head_weight);
        layer_work_coefficients = std::move(prepared);
        head_work_per_row = prepared_head_work;
        work_coefficients_model = &model;
    }

    GpuSegmentContinuation_vulkan::Implementation& acquire()
    {
        const size_t index = work.size();
        if (index == slots.size())
            slots.push_back(std::make_unique<GpuSegmentContinuation_vulkan::Implementation>());
        auto& result = *slots[index];
        // Publish the slot before assigning any target/cache owners, so an
        // allocation exception while preparing it is covered by retirement.
        work.push_back(&result);
        return result;
    }

    void retire() noexcept
    {
        for (auto* pending : work) pending->retire();
        work.clear();
        bank_storage.clear();
    }
};
#endif

GpuSegment_vulkan::Implementation::Implementation() = default;
GpuSegment_vulkan::Implementation::~Implementation() = default;

GpuSegmentContinuation_vulkan::GpuSegmentContinuation_vulkan() : d(new Implementation)
{
}

GpuSegmentContinuation_vulkan::~GpuSegmentContinuation_vulkan()
{
    abort();
#if NCNN_MOE_WITH_VULKAN
    if (d->target && d->target->context)
    {
        // Pipeline owners must outlive the allocator lock: their deleters
        // acquire it themselves. Raw recorded operands are freed under it.
        const auto target = d->target;
        const auto add = d->previous.add;
        const auto shared = d->previous.shared;
        const std::lock_guard<std::mutex> lock(target->context->command_mutex());
        d.reset();
    }
#endif
}

const std::shared_ptr<VulkanContext>& GpuSegmentContinuation_vulkan::context() const noexcept
{
    return d->target->context;
}

void GpuSegmentContinuation_vulkan::mark_submit_attempted() noexcept
{
    if (d->recorded && !d->failed && !d->published) d->attempted = true;
}

void GpuSegmentContinuation_vulkan::mark_completed() noexcept
{
    if (d->attempted && !d->failed && !d->published) d->completed = true;
}

void GpuSegmentContinuation_vulkan::abort() noexcept
{
    if (!d || d->published || (d->failed && !d->attempted)) return;
#if NCNN_MOE_WITH_VULKAN
    if (d->attempted)
    {
        for (auto& work : d->active_attention_work()) work->mark_submit_failed();
        for (auto& work : d->delta_work) work->mark_submit_failed();
        for (auto& work : d->mla_work) work->abort();
        // QSA can append into a shared allocation before metadata publication.
        // A submitted, discarded transaction cannot leave it reusable as old KV.
        if (!d->qsa_work.empty())
            for (auto* cache : d->caches) cache->vulkan_attention_state_unknown = true;
    }
#endif
    d->failed = true;
}

#if NCNN_MOE_WITH_VULKAN
bool GpuSegmentContinuation_vulkan::record(const ncnn::VkMat& combined, ncnn::VkCompute& command, const ncnn::Option& option)
{
    if (!d->previous.context || d->previous.combined || combined.empty()
        || activation_variant(option) != 0) return false;
    return d->record_work(&combined, command);
}
#endif

std::shared_ptr<GpuSegmentContinuation_vulkan> GpuSegment_vulkan::prepare_continuation(std::span<const uint64_t> positions,
                                                                                       std::span<LayerCache* const> caches,
                                                                                       const ExpertDispatchOptions& options,
                                                                                       const GpuSegmentWorkspace_vulkan& workspace,
                                                                                       std::string& error) const
{
#if NCNN_MOE_WITH_VULKAN
    if (d->head || !workspace.active() || workspace.d->combined || workspace.d->context != d->context
        || positions.empty() || positions.size() != caches.size() || positions.size() != workspace.d->residual->rows()
        || positions.size() > static_cast<size_t>(std::numeric_limits<int>::max())
        || std::any_of(caches.begin(), caches.end(), [](auto* cache) { return cache == nullptr; }))
    {
        error = "invalid Vulkan continuation inputs";
        return {};
    }
    auto result = std::shared_ptr<GpuSegmentContinuation_vulkan>(new GpuSegmentContinuation_vulkan);
    auto& work = *result->d;
    work.target = d;
    work.previous = *workspace.d;
    work.positions.assign(positions.begin(), positions.end());
    work.caches.assign(caches.begin(), caches.end());
    work.options = options;
    work.selection_bias.assign(options.selection_bias.begin(), options.selection_bias.end());
    work.explicit_ids.assign(options.explicit_expert_ids.begin(), options.explicit_expert_ids.end());
    work.options.selection_bias = work.selection_bias;
    work.options.explicit_expert_ids = work.explicit_ids;
    return result;
#else
    (void)positions;
    (void)caches;
    (void)options;
    (void)workspace;
    error = "Vulkan continuations are unavailable in this build";
    return {};
#endif
}

std::shared_ptr<GpuSegmentContinuation_vulkan> GpuSegment_vulkan::prepare_head_continuation(bool produce_logits, bool last_row_only,
                                                                                            const GpuSegmentWorkspace_vulkan& workspace,
                                                                                            std::string& error) const
{
#if NCNN_MOE_WITH_VULKAN
    if (!d->head || !workspace.active() || workspace.d->combined || workspace.d->context != d->context)
    {
        error = "invalid Vulkan head continuation inputs";
        return {};
    }
    auto result = std::shared_ptr<GpuSegmentContinuation_vulkan>(new GpuSegmentContinuation_vulkan);
    result->d->target = d;
    if (!workspace.d->host_output) workspace.d->host_output = std::make_shared<VulkanHostOutput>();
    result->d->host_output = workspace.d->host_output;
    result->d->host_output_slot_size = workspace.d->host_output_slot_size;
    result->d->previous = *workspace.d;
    result->d->produce_logits = produce_logits;
    result->d->last_row_only = last_row_only;
    return result;
#else
    (void)produce_logits;
    (void)last_row_only;
    (void)workspace;
    error = "Vulkan continuations are unavailable in this build";
    return {};
#endif
}

bool GpuSegmentContinuation_vulkan::publish(ExpertDispatchPlan& routes, GpuSegmentWorkspace_vulkan& workspace, std::string& error)
{
#if NCNN_MOE_WITH_VULKAN
    if (!d->completed || d->failed || d->published || !d->target || d->target->head || !d->same_workspace(*workspace.d))
    {
        error = d->error.empty() ? "Vulkan continuation is not publishable" : d->error;
        return false;
    }
    const auto& target = *d->target;
    const std::lock_guard<std::mutex> lock(target.context->command_mutex());
    auto fail = [&](const char* reason) { error = reason; abort(); return false; };
    ActivationBuffer selected(d->positions.size(), d->selected_columns);
    ExpertDispatchPlan decoded;
    auto normalized = std::make_shared<DeviceTensor_vulkan>();
    auto residual = std::make_shared<DeviceTensor_vulkan>();
    std::shared_ptr<DeviceTensor_vulkan> dense_hidden;
    if (!target.dense)
    {
        if (!copy_staging_to_cpu_batch(d->selected_staging, selected)) return fail("cannot read compact Vulkan routes");
        const auto decoded_result = Router_vulkan::decode_selected(selected, d->options, decoded);
        if (!decoded_result)
        {
            error = decoded_result.error().message;
            abort();
            return false;
        }
    }
    if (!normalized->assign_completed(d->normalized, target.context) || !residual->assign_completed(d->residual, target.context))
        return fail("cannot publish completed Vulkan segment activations");
    if (target.dense)
    {
        dense_hidden = std::make_shared<DeviceTensor_vulkan>();
        if (!dense_hidden->assign_completed(d->dense_output, target.context))
            return fail("cannot publish completed Vulkan dense layer");
    }
    for (auto& work : d->active_attention_work())
        if (!work->commit()) return fail("Standard Attention cache publication failed");
    for (auto& work : d->delta_work)
        if (!work->commit()) return fail("GatedDeltaNet state publication failed");
    for (auto& work : d->mla_work)
        if (!work->commit()) return fail("expanded-value latent attention cache publication failed");
    for (auto& work : d->qsa_work)
        if (!work->commit()) return fail("QSA cache publication failed");
    auto& current = *workspace.d;
    current.context = target.context;
    current.option = target.option;
    current.add = target.add;
    current.shared = target.shared;
    current.normalized = target.dense ? std::shared_ptr<const DeviceTensor_vulkan>(dense_hidden)
                                      : std::shared_ptr<const DeviceTensor_vulkan>(normalized);
    current.residual = std::move(residual);
    current.combined.reset();
    current.dense_ready = target.dense != nullptr;
    routes = std::move(decoded);
    auto& counters = target.context->runtime_state();
    counters.dispatches += d->ffn_statistics.dispatches + 3;
    counters.int4_dispatches += d->ffn_statistics.int4_dispatches;
    counters.shared_expert_swiglu_fusions += d->ffn_statistics.shared_expert_swiglu_fusions;
    counters.bfloat16_cooperative_matrix_dispatches += d->ffn_statistics.bfloat16_cooperative_matrix_dispatches;
    counters.compute_submissions += d->standalone;
    ++counters.resident_graph_submissions;
    counters.batch_uploads += d->uploaded;
    ++counters.batch_downloads;
    if (d->uploaded) counters.resident_graph_activation_upload_bytes += d->positions.size() * target.hidden * sizeof(float);
    counters.resident_graph_route_download_bytes += selected.bytes().size();
    d->published = true;
    return true;
#else
    (void)routes;
    (void)workspace;
    error = "Vulkan continuations are unavailable in this build";
    return false;
#endif
}

bool GpuSegmentContinuation_vulkan::publish_head(ActivationBuffer& logits, GpuSegmentWorkspace_vulkan& workspace, std::string& error)
{
#if NCNN_MOE_WITH_VULKAN
    if (!d->completed || d->failed || d->published || !d->target || !d->target->head || !d->same_workspace(*workspace.d))
    {
        error = d->error.empty() ? "Vulkan head continuation is not publishable" : d->error;
        return false;
    }
    const auto& target = *d->target;
    const std::lock_guard<std::mutex> lock(target.context->command_mutex());
    if (d->produce_logits)
    {
        if (!VulkanHostOutput::complete(d->selected_staging, d->pending_logits))
        {
            error = "cannot read Vulkan logits";
            abort();
            return false;
        }
    }
    auto& counters = target.context->runtime_state();
    counters.dispatches += d->ffn_statistics.dispatches + (d->produce_logits ? (d->fused_head ? 1 : 2) : 0);
    counters.rms_norm_linear_fusions += d->fused_head;
    counters.int4_dispatches += d->ffn_statistics.int4_dispatches + (d->produce_logits && target.lm_head.qnk && target.lm_head.qnk->dtype() == DType::Int4 ? 1 : 0);
    counters.shared_expert_swiglu_fusions += d->ffn_statistics.shared_expert_swiglu_fusions;
    counters.bfloat16_cooperative_matrix_dispatches += d->ffn_statistics.bfloat16_cooperative_matrix_dispatches;
    counters.compute_submissions += d->standalone;
    ++counters.resident_graph_submissions;
    counters.batch_downloads += d->produce_logits;
    if (d->produce_logits)
    {
        logits.swap(d->pending_logits);
        d->pending_logits.clear();
    }
    auto& current = *workspace.d;
    current.normalized.reset();
    current.residual.reset();
    current.combined.reset();
    current.shared.reset();
    current.add.reset();
    current.context.reset();
    d->published = true;
    return true;
#else
    (void)logits;
    (void)workspace;
    error = "Vulkan continuations are unavailable in this build";
    return false;
#endif
}

bool GpuSegment_vulkan::forward_resident(const CompiledModel& model, const ActivationBuffer& initial_hidden,
                                         std::span<const uint64_t> positions, std::span<LayerCache* const> caches,
                                         std::span<const ExpertDispatchOptions> options,
                                         bool produce_logits, bool last_row_only, ActivationBuffer& logits, std::string& error,
                                         GpuSegmentWorkspace_vulkan* workspace, uint32_t layer_limit)
{
#if NCNN_MOE_WITH_VULKAN
    const size_t layers = model.net.layer_plans.size();
    const size_t rows = positions.size();
    const size_t limit = layer_limit == 0 ? layers : std::min(layers, static_cast<size_t>(layer_limit));
    if (layers == 0 || limit == 0 || rows == 0 || rows > static_cast<size_t>(std::numeric_limits<int>::max())
        || layers > std::numeric_limits<size_t>::max() / rows || caches.size() != layers * rows
        || (!options.empty() && options.size() != layers)
        || (!model.gpu_expert_banks.empty() && model.gpu_expert_banks.size() != layers)
        || model.gpu_segments.size() != layers || (!model.gpu_head && limit == layers)
        || model.resident_submission_rows == 0
        || (limit < layers && produce_logits)
        || initial_hidden.dtype() != DType::Float32 || initial_hidden.rows() != rows
        || initial_hidden.columns() != model.descriptor.hidden_size
        || std::any_of(caches.begin(), caches.end(), [](auto* cache) { return cache == nullptr || cache->vulkan_attention_state_unknown; }))
    {
        error = "invalid resident graph chain inputs";
        return false;
    }
    const auto context = model.gpu_head ? model.gpu_head->context() : model.gpu_segments.front()->context();
    for (size_t layer = 0; layer < limit; ++layer)
    {
        const auto& segment = model.gpu_segments[layer];
        if (!context || !segment || segment->context() != context)
        {
            error = "resident graph chain has incompatible devices";
            return false;
        }
        const bool dense = model.net.layer_plans[layer].dense_ffn.down_weight != invalid_tensor_handle;
        if (dense)
        {
            const bool attention_ready = segment->d->attention || segment->d->delta || segment->d->qsa;
            if (!segment->d->dense || !attention_ready || segment->d->dense->context() != context
                || (!model.gpu_expert_banks.empty() && model.gpu_expert_banks[layer]))
            {
                error = "resident graph chain has incompatible dense operators";
                return false;
            }
        }
        else if (segment->d->dense || options.size() != layers || model.gpu_expert_banks.size() != layers
                 || !model.gpu_expert_banks[layer] || model.gpu_expert_banks[layer]->context() != context
                 || options[layer].expert_count != model.gpu_expert_banks[layer]->description().expert_count
                 || options[layer].top_k == 0 || options[layer].top_k > options[layer].expert_count)
        {
            error = "resident graph chain has incompatible bank or routing metadata";
            return false;
        }
    }
    const uint32_t chunk_layers = std::max<uint32_t>(1, std::min<uint32_t>(resident_chunk_layers(model, static_cast<uint32_t>(rows)), static_cast<uint32_t>(limit)));
    std::optional<GpuSegmentWorkspace_vulkan> local_workspace;
    if (!workspace) local_workspace.emplace();
    auto& output_workspace = workspace ? *workspace : *local_workspace;
    if (!output_workspace.d->host_output) output_workspace.d->host_output = std::make_shared<VulkanHostOutput>();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    auto& head_implementation = *model.gpu_head->d;
    if (!head_implementation.resident_workspace)
        head_implementation.resident_workspace = std::make_unique<Implementation::ResidentWorkspace>();
    auto& pooled = *head_implementation.resident_workspace;
    ncnn::VkMat hidden;
    ncnn::VkMat validation_upload, validation, validation_download;
    auto& work = pooled.work;
    auto& bank_storage = pooled.bank_storage;
    // The original chunk remains the allocation/retirement unit. Internal
    // submissions share one exclusively leased queue and retain every operand.
    VulkanComputeBatch batches(*context);
    pooled.prepare_work_coefficients(model);
    auto& layer_work = pooled.layer_work;
    layer_work.resize(limit);
    const double head_work = produce_logits ? pooled.head_work_per_row * (last_row_only ? 1.0 : rows) : 0;
    double total_work = head_work;
    for (size_t layer = 0; layer < limit; ++layer)
    {
        const auto& coefficients = pooled.layer_work_coefficients[layer];
        auto& estimate = layer_work[layer];
        estimate.attention_router = std::max(1.0, rows * coefficients.attention_router_per_row);
        const uint32_t selected = model.gpu_segments[layer]->d->dense ? 0 : options[layer].top_k;
        estimate.ffn = std::max(1.0, rows * (selected * coefficients.routed_expert_per_selection_row + coefficients.fixed_ffn_per_row));
        total_work += estimate.attention_router + estimate.ffn;
    }
    GpuSubmissionBudget budget(total_work);
    bool successful = false, submitted = false, chunk_attempted = false;
    const auto cleanup = [&](void*) {
        if (!successful)
        {
            (void)batches.reset(); // Drain first, before marking/freeing any pending state.
            if (chunk_attempted)
                for (auto& pending : work)
                {
                    for (auto& attention : pending->active_attention_work()) attention->mark_submit_failed();
                    for (auto& delta : pending->delta_work) delta->mark_submit_failed();
                    for (auto& mla : pending->mla_work) mla->abort();
                }
            // A partially executed chain cannot be replayed against the old
            // Session position, including layers committed by an earlier chunk.
            if (submitted)
                for (auto* cache : caches) cache->vulkan_attention_state_unknown = true;
            // Also run on exceptions: never leave target/cache references in
            // the head-owned pool, and never move GPU operand cleanup to a later token.
            pooled.retire();
        }
    };
    std::unique_ptr<void, decltype(cleanup)> guard(reinterpret_cast<void*>(1), cleanup);
    auto fail = [&](const std::string& reason) { error = reason; return false; };
    for (size_t first = 0; first < limit; first += chunk_layers)
    {
        auto* recording = batches.next();
        if (!recording) return fail("cannot acquire ordered resident graph command");
        validation_upload.release();
        validation.release();
        validation_download.release();
        chunk_attempted = false;
        // Cross-command access metadata must be rearmed only after a fence.
        // Fresh intra-command values keep their producer's access state.
        if (!hidden.empty())
        {
            hidden.data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            hidden.data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
        const size_t end = std::min(limit, first + chunk_layers);
        const bool routed = std::any_of(model.gpu_segments.begin() + first, model.gpu_segments.begin() + end,
                                        [](const auto& segment) { return !segment->d->dense; });
        const auto& option = model.gpu_segments[first]->d->option;
        if (routed)
        {
            const uint32_t zero = 0;
            if (!fill_staging_values(&zero, 1, sizeof(zero), validation_upload, context->staging_allocator())
                || !record_mapped_upload(validation_upload, validation, *recording, option))
                return fail("cannot initialize resident Expert bank validation");
        }
        const auto flush_before = [&](double next_work) {
            if (!budget.would_flush_before(next_work)) return true;
            submitted = chunk_attempted = true;
            if (batches.submit() != 0) return fail("resident graph asynchronous submit failed");
            budget.submitted();
            recording = batches.next();
            if (!recording) return fail("cannot acquire next resident graph command");
            return true;
        };
        for (size_t layer = first; layer < end; ++layer)
        {
            if (!flush_before(layer_work[layer].attention_router)) return false;
            auto& current = pooled.acquire();
            current.target = model.gpu_segments[layer]->d;
            current.positions.assign(positions.begin(), positions.end());
            current.caches.assign(caches.begin() + layer * rows, caches.begin() + (layer + 1) * rows);
            if (!current.target->dense) current.options = options[layer];
            current.initial_hidden = &initial_hidden;
            if (!current.record_layer(nullptr, *recording, hidden.empty() ? nullptr : &hidden, false))
                return fail(current.error);
            budget.record(layer_work[layer].attention_router);
            // Submit completed record-only stages before a large successor.
            // All operands and producer access metadata survive until the
            // original chunk fence, including Router IDs consumed below.
            if (!flush_before(layer_work[layer].ffn)) return false;
            auto& command = *recording;
            auto& target = *current.target;
            if (target.dense)
            {
                ncnn::VkMat output;
                if (!target.dense->record(current.normalized, output, command, current.storage, current.ffn_statistics))
                    return fail("cannot record resident dense FFN");
                std::vector<ncnn::VkMat> added(1);
                if (target.add->forward({current.residual, output}, added, command, target.option) != 0 || added[0].empty())
                    return fail("cannot record resident dense FFN residual");
                hidden = added[0];
                current.storage.push_back(hidden);
                ++current.ffn_statistics.dispatches;
            }
            else
            {
                ncnn::VkMat shared;
                const ncnn::VkMat* shared_input = nullptr;
                if (target.shared)
                {
                    if (!target.shared->record(current.normalized, shared, command, current.storage, current.ffn_statistics))
                        return fail("cannot record resident shared Expert");
                    current.storage.push_back(shared);
                    shared_input = &shared;
                }
                ncnn::VkMat combined;
                if (!model.gpu_expert_banks[layer]->record(current.normalized, current.router_work.selected,
                                                           options[layer].top_k, combined, validation, command, bank_storage,
                                                           shared_input, &current.residual))
                    return fail("cannot record resident indexed Expert bank");
                hidden = combined;
            }
            budget.record(layer_work[layer].ffn);
        }
        if (end == limit && limit == layers)
        {
            if (produce_logits && !flush_before(head_work)) return false;
            auto& head = pooled.acquire();
            head.target = model.gpu_head->d;
            head.host_output = output_workspace.d->host_output;
            head.host_output_slot_size = output_workspace.d->host_output_slot_size;
            head.produce_logits = produce_logits;
            head.last_row_only = last_row_only;
            if (!head.record_head(nullptr, *recording, &hidden)) return fail(head.error);
            if (produce_logits) budget.record(head_work);
        }
        auto& command = *recording;
        if (routed && (!prepare_staging_batch(validation_download, 1, 1, context->staging_allocator()) || !record_staging_download(validation, 1, 1, validation_download, command, context->device(), option)))
            return fail("cannot record resident Expert bank validation download");
        submitted = chunk_attempted = true;
        if (batches.submit() != 0)
            return fail("resident graph submit failed");
        budget.submitted();
        const size_t chunk_submissions = batches.submission_count();
        // reset drains submitted commands and clears the batch counters.
        // Do it before readback/publication so a reset error still aborts the chunk.
        if (batches.reset() != 0) return fail("resident graph fence wait failed");
        if (routed)
        {
            ActivationBuffer checked(1, 1);
            if (!copy_staging_to_cpu_batch(validation_download, checked)
                || std::bit_cast<uint32_t>(checked.row(0)[0]) != 0)
                return fail("resident Expert bank rejected invalid GPU routing data");
        }
        ActivationBuffer completed;
        if (end == limit && limit == layers && produce_logits)
        {
            auto& head = *work.back();
            if (!VulkanHostOutput::complete(head.selected_staging, head.pending_logits))
                return fail("cannot read resident graph logits");
            completed.swap(head.pending_logits);
        }
        for (auto& pending : work)
        {
            for (auto& attention : pending->active_attention_work())
                if (!attention->commit()) return fail("resident graph Attention cache publication failed");
            for (auto& delta : pending->delta_work)
                if (!delta->commit()) return fail("resident graph GatedDeltaNet publication failed");
            for (auto& mla : pending->mla_work)
                if (!mla->commit()) return fail("resident graph expanded-value latent attention publication failed");
            for (auto& qsa : pending->qsa_work)
                if (!qsa->commit()) return fail("resident graph QSA publication failed");
            auto& counters = context->runtime_state();
            const auto& stats = pending->ffn_statistics;
            counters.dispatches += stats.dispatches + (pending->target->head ? (produce_logits ? (pending->fused_head ? 1 : 2) : 0) : (pending->target->dense ? 1 : 6));
            counters.int4_dispatches += stats.int4_dispatches
                                        + (pending->target->head && produce_logits && pending->target->lm_head.qnk
                                                   && pending->target->lm_head.qnk->dtype() == DType::Int4
                                               ? 1
                                               : 0);
            counters.shared_expert_swiglu_fusions += stats.shared_expert_swiglu_fusions;
            counters.bfloat16_cooperative_matrix_dispatches += stats.bfloat16_cooperative_matrix_dispatches;
            counters.rms_norm_linear_fusions += pending->fused_head;
        }
        auto& counters = context->runtime_state();
        counters.compute_submissions += chunk_submissions;
        counters.resident_graph_submissions += chunk_submissions;
        counters.batch_uploads += (first == 0 ? 1 : 0) + (routed ? 1 : 0);
        counters.batch_downloads += (end == limit && limit == layers && produce_logits ? 1 : 0) + (routed ? 1 : 0);
        if (first == 0) counters.resident_graph_activation_upload_bytes += rows * model.descriptor.hidden_size * sizeof(float);
        if (end == limit && limit == layers && produce_logits) logits.swap(completed);
        if (end == limit && limit < layers)
        {
            auto completed_hidden = std::make_shared<DeviceTensor_vulkan>();
            if (!completed_hidden->assign_completed(hidden, context))
                return fail("cannot publish resident graph prefix hidden");
            output_workspace.d->context = context;
            output_workspace.d->option = option;
            output_workspace.d->normalized = completed_hidden;
            output_workspace.d->residual = completed_hidden;
            output_workspace.d->dense_ready = true;
        }
        chunk_attempted = false;
        // Release completed private operands before the next recording chunk.
        pooled.retire();
    }
    if (!produce_logits) logits.clear();
    successful = true;
    return true;
#else
    (void)model;
    (void)initial_hidden;
    (void)positions;
    (void)caches;
    (void)options;
    (void)produce_logits;
    (void)last_row_only;
    (void)logits;
    (void)workspace;
    error = "resident graphs are unavailable in this build";
    return false;
#endif
}

bool GpuSegment_vulkan::forward(const ActivationBuffer& initial_hidden, std::span<const uint64_t> positions,
                                std::span<LayerCache* const> caches, const ExpertDispatchOptions& options,
                                ExpertDispatchPlan& routes, GpuSegmentWorkspace_vulkan& workspace, std::string& error) const
{
#if NCNN_MOE_WITH_VULKAN
    if (d->head || !d->context || positions.empty() || positions.size() != caches.size()
        || positions.size() > static_cast<size_t>(std::numeric_limits<int>::max())
        || std::any_of(caches.begin(), caches.end(), [](auto* cache) { return cache == nullptr; })
        || (workspace.d->context && workspace.d->context != d->context)
        || (!workspace.d->context && (initial_hidden.dtype() != DType::Float32 || initial_hidden.rows() != positions.size() || initial_hidden.columns() != d->hidden)))
    {
        error = "invalid Vulkan segment inputs";
        return false;
    }
    GpuSegmentContinuation_vulkan pending;
    auto& work = *pending.d;
    work.target = d;
    work.previous = *workspace.d;
    work.positions.assign(positions.begin(), positions.end());
    work.caches.assign(caches.begin(), caches.end());
    work.options = options;
    work.initial_hidden = &initial_hidden;
    work.standalone = true;
    auto lease = d->context->acquire_transfer_slot();
    {
        auto& slot = lease.slot();
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        if (!reset_segment_command(slot))
        {
            error = "cannot reset Vulkan segment command";
            return false;
        }
        auto& command = *slot.command;
        if (!work.record_work(nullptr, command))
        {
            (void)command.reset();
            error = work.error;
            return false;
        }
        pending.mark_submit_attempted();
        if (submit_compute_and_wait(command, d->context->device()) != 0)
        {
            (void)command.reset();
            pending.abort();
            error = "Vulkan segment submit or fence wait failed";
            return false;
        }
        pending.mark_completed();
    }
    return pending.publish(routes, workspace, error);
#else
    (void)initial_hidden;
    (void)positions;
    (void)caches;
    (void)options;
    (void)routes;
    (void)workspace;
    error = "Vulkan segments are unavailable in this build";
    return false;
#endif
}

bool GpuSegment_vulkan::defer_combine(std::shared_ptr<const DeviceTensor_vulkan> combined, GpuSegmentWorkspace_vulkan& workspace)
{
#if NCNN_MOE_WITH_VULKAN
    auto& current = *workspace.d;
    if (!workspace.active() || current.combined || !combined || combined->empty() || !combined->value()
        || combined->vulkan_context() != current.context || combined->rows() != current.residual->rows()
        || combined->columns() != current.residual->columns()) return false;
    current.combined = std::move(combined);
    return true;
#else
    (void)combined;
    (void)workspace;
    return false;
#endif
}

bool GpuSegment_vulkan::materialize_hidden(GpuSegmentWorkspace_vulkan& workspace, ActivationBuffer& hidden, std::string& error)
{
#if NCNN_MOE_WITH_VULKAN
    auto& current = *workspace.d;
    if (!workspace.active())
    {
        error = "no resident Vulkan hidden at CPU boundary";
        return false;
    }
    const auto context = current.context;
    const auto add_owner = current.add;
    const auto shared_owner = current.shared;
    auto lease = context->acquire_transfer_slot();
    auto& slot = lease.slot();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    if (!reset_segment_command(slot))
    {
        error = "cannot reset Vulkan materialization command";
        return false;
    }
    auto& command = *slot.command;
    std::vector<ncnn::VkMat> storage;
    ncnn::VkMat output;
    FfnRecordStatistics_vulkan statistics;
    bool successful = false;
    const auto cleanup = [&](void*) { if (!successful) (void)command.reset(); };
    std::unique_ptr<void, decltype(cleanup)> guard(reinterpret_cast<void*>(1), cleanup);
    const size_t rows = current.dense_ready ? current.normalized->rows() : current.residual->rows();
    const uint32_t columns = current.dense_ready ? current.normalized->columns() : current.residual->columns();
    if (current.dense_ready)
        output = *current.normalized->value();
    if ((!current.dense_ready && !current.record_hidden(output, command, storage, statistics))
        || !prepare_staging_batch(slot.download, rows, columns, slot.staging_allocator)
        || !record_staging_download(output, rows, columns, slot.download, command, context->device(), current.option)
        || submit_compute_and_wait(command, context->device()) != 0)
    {
        error = "Vulkan hidden materialization failed";
        return false;
    }
    ActivationBuffer completed(rows, columns);
    if (!copy_staging_to_cpu_batch(slot.download, completed))
    {
        error = "cannot read materialized Vulkan hidden";
        return false;
    }
    auto& counters = context->runtime_state();
    counters.dispatches += statistics.dispatches;
    counters.int4_dispatches += statistics.int4_dispatches;
    counters.shared_expert_swiglu_fusions += statistics.shared_expert_swiglu_fusions;
    counters.bfloat16_cooperative_matrix_dispatches += statistics.bfloat16_cooperative_matrix_dispatches;
    ++counters.compute_submissions;
    ++counters.resident_graph_submissions;
    ++counters.batch_downloads;
    counters.resident_graph_activation_download_bytes += completed.bytes().size();
    hidden.swap(completed);
    current.normalized.reset();
    current.residual.reset();
    current.combined.reset();
    current.dense_ready = false;
    current.shared.reset();
    current.add.reset();
    current.context.reset();
    successful = true;
    return true;
#else
    (void)workspace;
    (void)hidden;
    error = "Vulkan segments are unavailable in this build";
    return false;
#endif
}

bool GpuSegment_vulkan::finish(bool produce_logits, bool last_row_only, ActivationBuffer& hidden, ActivationBuffer& logits,
                               GpuSegmentWorkspace_vulkan& workspace, std::string& error) const
{
#if NCNN_MOE_WITH_VULKAN
    (void)hidden;
    if (!d->head || !workspace.active() || workspace.d->context != d->context)
    {
        error = "Vulkan head has no compatible resident segment";
        return false;
    }
    GpuSegmentContinuation_vulkan pending;
    auto& work = *pending.d;
    work.target = d;
    if (!workspace.d->host_output) workspace.d->host_output = std::make_shared<VulkanHostOutput>();
    work.host_output = workspace.d->host_output;
    work.host_output_slot_size = workspace.d->host_output_slot_size;
    work.previous = *workspace.d;
    work.produce_logits = produce_logits;
    work.last_row_only = last_row_only;
    work.standalone = true;
    auto lease = d->context->acquire_transfer_slot();
    {
        auto& slot = lease.slot();
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        if (!reset_segment_command(slot))
        {
            error = "cannot reset Vulkan head command";
            return false;
        }
        auto& command = *slot.command;
        if (!work.record_work(nullptr, command))
        {
            (void)command.reset();
            error = work.error;
            return false;
        }
        pending.mark_submit_attempted();
        if (submit_compute_and_wait(command, d->context->device()) != 0)
        {
            (void)command.reset();
            pending.abort();
            error = "Vulkan head submit or fence wait failed";
            return false;
        }
        pending.mark_completed();
    }
    return pending.publish_head(logits, workspace, error);
#else
    (void)produce_logits;
    (void)last_row_only;
    (void)hidden;
    (void)logits;
    (void)workspace;
    error = "Vulkan segments are unavailable in this build";
    return false;
#endif
}

} // namespace moe
} // namespace ncnn
