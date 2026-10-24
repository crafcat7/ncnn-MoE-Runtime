#include "qwenlayer_vulkan.h"
#include "gatedresidual_vulkan.h"
#include "gateddeltanet_vulkan.h"
#include "qsaattention_vulkan.h"
#include "qwen_vulkan_internal.h"
#include "router_vulkan.h"
#include "rmsnorm_vulkan.h"
#include "moecombine_vulkan.h"
#include "linear.h"
#include "graph/compiledmodel.h"
#include "kernels/statecache.h"
#include "kernels/qnk.h"
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#if NCNN_MOE_WITH_VULKAN
#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif

namespace ncnn {
namespace moe {
class PleState_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::VkMat convolution;
    uint32_t slot = 0;
    bool valid = true;
#endif
};
PleState_vulkan::PleState_vulkan() : d(new Implementation)
{
}
PleState_vulkan::~PleState_vulkan()
{
#if NCNN_MOE_WITH_VULKAN
    if (d->context)
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        d->convolution.release();
    }
#endif
}
uint64_t PleState_vulkan::allocated_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->convolution.empty() ? 0 : d->convolution.total() * d->convolution.elemsize;
#else
    return 0;
#endif
}

class QwenLayerWorkspace_vulkan::Implementation
{
public:
    std::shared_ptr<const DeviceTensor_vulkan> normalized;
    ActivationBuffer selected;
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<GatedResidual_vulkan> post;
    std::shared_ptr<MoeCombine_vulkan> combiner;
    std::shared_ptr<MoeCombinePending_vulkan> pending;
    uint64_t pending_dispatches = 0;
    ncnn::VkMat residual, injection;
    bool record_hidden(ncnn::VkMat& result, ncnn::VkCompute& cmd, std::vector<ncnn::VkMat>& storage, uint64_t& uploads)
    {
        if (!pending || !post || !combiner) return false;
        ncnn::VkMat combined;
        qwen_prepare_read(residual);
        qwen_prepare_read(injection);
        if (!combiner->record(*pending, combined, cmd, storage, uploads)
            || !post->record_post(combined, residual, injection, result, cmd)) return false;
        storage.push_back(combined);
        return true;
    }
    void clear_buffers()
    {
        pending_dispatches = 0;
        normalized.reset();
        pending.reset();
        residual.release();
        injection.release();
    }
#endif
};
QwenLayerWorkspace_vulkan::QwenLayerWorkspace_vulkan() : d(new Implementation)
{
}
QwenLayerWorkspace_vulkan::~QwenLayerWorkspace_vulkan()
{ reset(); }
void QwenLayerWorkspace_vulkan::reset()
{
#if NCNN_MOE_WITH_VULKAN
    const auto context = d->context;
    const auto post = d->post;
    const auto combiner = d->combiner;
    if (context)
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        d->clear_buffers();
    }
    d->post.reset();
    d->combiner.reset();
    d->context.reset();
#endif
}
bool QwenLayerWorkspace_vulkan::active() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->context && d->post && !d->residual.empty();
#else
    return false;
#endif
}
uint64_t QwenLayerWorkspace_vulkan::allocated_bytes() const noexcept
{ return d->selected.allocated_bytes(); }
const std::shared_ptr<const DeviceTensor_vulkan>& QwenLayerWorkspace_vulkan::device_input() const noexcept
{ return d->normalized; }

class QwenLayer_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    const WeightStore* weights = nullptr;
    PleBlockPlan ple;
    uint32_t hidden = 0, copies = 0;
    float epsilon = 0, offset = 0;
    std::shared_ptr<GatedResidual_vulkan> attention_hyper, ffn_hyper;
    std::shared_ptr<GatedDeltaNet_vulkan> delta;
    std::shared_ptr<QsaAttention_vulkan> qsa;
    std::shared_ptr<Router_vulkan> router;
    std::shared_ptr<MoeCombine_vulkan> combiner;
    std::shared_ptr<QnkLinear_vulkan> ple_key, ple_value, lm_head;
    std::shared_ptr<RmsNorm_vulkan> final_norm;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    ncnn::VkMat key_norm, query_norm, conv_norm, convolution_weight;
    uint64_t bytes = 0;
    void release_buffers()
    {
        key_norm.release();
        query_norm.release();
        conv_norm.release();
        convolution_weight.release();
    }
#endif
};
QwenLayer_vulkan::QwenLayer_vulkan() : d(new Implementation)
{
}
QwenLayer_vulkan::~QwenLayer_vulkan()
{
#if NCNN_MOE_WITH_VULKAN
    if (d->context)
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        d->release_buffers();
    }
#endif
}
uint64_t QwenLayer_vulkan::device_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->bytes;
#else
    return 0;
#endif
}

std::shared_ptr<QwenLayer_vulkan> QwenLayer_vulkan::create(const CompiledModel& model, const CompiledLayerPlan& layer)
{
#if NCNN_MOE_WITH_VULKAN
    auto result = std::shared_ptr<QwenLayer_vulkan>(new QwenLayer_vulkan);
    auto& impl = *result->d;
    impl.weights = &model.weights;
    impl.ple = layer.ple;
    impl.hidden = model.descriptor.hidden_size;
    impl.copies = model.descriptor.hyper_connection_multiplier;
    impl.epsilon = model.descriptor.norm_epsilon;
    impl.offset = model.descriptor.norm_weight_offset;
    impl.attention_hyper = GatedResidual_vulkan::create(model.weights, layer.attention_gated_residual,
                                                        model.descriptor, layer.vulkan_device_index, model.vulkan_runtime, model.opt.optimization_flags);
    impl.ffn_hyper = GatedResidual_vulkan::create(model.weights, layer.ffn_gated_residual,
                                                  model.descriptor, layer.vulkan_device_index, model.vulkan_runtime, model.opt.optimization_flags);
    if (!impl.attention_hyper || !impl.ffn_hyper) return {};
    impl.context = impl.attention_hyper->context();
    impl.opt = impl.attention_hyper->option();
    if (layer.attention.kind == AttentionKind::GatedDeltaNet)
        impl.delta = model.operators.at(layer.attention.gated_delta_vulkan_operator).gated_delta;
    else
        impl.qsa = model.operators.at(layer.attention.vulkan_attention_operator).qsa_attention;
    if ((!impl.delta && !impl.qsa) || (impl.delta && impl.qsa)) return {};
    if ((impl.delta && impl.delta->vulkan_context() != impl.context) || (impl.qsa && impl.qsa->vulkan_context() != impl.context)) return {};
    impl.router = Router_vulkan::create(model.weights.at(layer.moe.router_weight),
                                        layer.moe.router_bias == invalid_tensor_handle ? nullptr : &model.weights.at(layer.moe.router_bias),
                                        layer.moe.router_selection_bias == invalid_tensor_handle ? nullptr : &model.weights.at(layer.moe.router_selection_bias),
                                        layer.vulkan_device_index, model.vulkan_runtime, model.opt.optimization_flags);
    impl.combiner = MoeCombine_vulkan::create(impl.context);
    if (!impl.router || !impl.combiner) return {};
    impl.bytes = impl.attention_hyper->device_bytes() + impl.ffn_hyper->device_bytes();
    if (impl.ple.enabled())
    {
        impl.ple_key = QnkLinear_vulkan::create(model.weights.at(impl.ple.key_weight), nullptr, layer.vulkan_device_index, model.vulkan_runtime, model.opt.optimization_flags);
        impl.ple_value = QnkLinear_vulkan::create(model.weights.at(impl.ple.value_weight), nullptr, layer.vulkan_device_index, model.vulkan_runtime, model.opt.optimization_flags);
        if (!impl.ple_key || !impl.ple_value || impl.ple.embedding_shards.empty()) return {};
        const uint32_t heads = (impl.ple.ngram_size - 1) * impl.ple.heads_per_ngram;
        if (heads == 0 || impl.ple.embedding_dimension % heads != 0) return {};
        for (auto handle : impl.ple.embedding_shards)
        {
            const auto& table = model.weights.at(handle);
            if (table.shape.size() != 2 || table.shape[1] != impl.ple.embedding_dimension / heads
                || (table.dtype != DType::IQ4NL && table.dtype != DType::Q8_0 && table.dtype != DType::BFloat16 && table.dtype != DType::Float32)) return {};
        }
        const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
        impl.pipeline = qwen_pipeline(impl.context, impl.opt);
        if (!impl.pipeline) return {};
        ncnn::VkCompute cmd(impl.context->device());
        if (!qwen_float_weight(model.weights.at(impl.ple.key_norm_weight), impl.key_norm, cmd, impl.opt)
            || !qwen_float_weight(model.weights.at(impl.ple.query_norm_weight), impl.query_norm, cmd, impl.opt)
            || !qwen_float_weight(model.weights.at(impl.ple.convolution_norm_weight), impl.conv_norm, cmd, impl.opt)
            || !qwen_float_weight(model.weights.at(impl.ple.convolution_weight), impl.convolution_weight, cmd, impl.opt)
            || submit_compute_and_wait(cmd, impl.context->device()) != 0) return {};
        impl.bytes += impl.ple_key->device_bytes() + impl.ple_value->device_bytes()
                      + impl.key_norm.total() * impl.key_norm.elemsize + impl.query_norm.total() * impl.query_norm.elemsize
                      + impl.conv_norm.total() * impl.conv_norm.elemsize + impl.convolution_weight.total() * impl.convolution_weight.elemsize;
    }
    return result;
#else
    (void)model;
    (void)layer;
    return {};
#endif
}

std::shared_ptr<QwenLayer_vulkan> QwenLayer_vulkan::create_head(const CompiledModel& model)
{
#if NCNN_MOE_WITH_VULKAN
    auto result = std::shared_ptr<QwenLayer_vulkan>(new QwenLayer_vulkan);
    auto& impl = *result->d;
    impl.attention_hyper = GatedResidual_vulkan::create(model.weights, model.gated_residual_head,
                                                        model.descriptor, model.opt.vulkan_device_index, model.vulkan_runtime, model.opt.optimization_flags);
    if (!impl.attention_hyper) return {};
    impl.context = impl.attention_hyper->context();
    impl.opt = impl.attention_hyper->option();
    impl.lm_head = model.operators.at_weight(model.lm_head_weight).qnk;
    if (!impl.lm_head) impl.lm_head = QnkLinear_vulkan::create(model.weights.at(model.lm_head_weight), nullptr,
                                                               model.opt.vulkan_device_index, model.vulkan_runtime, model.opt.optimization_flags);
    if (model.descriptor.final_norm == NormType::RmsNorm)
    {
        if (model.final_norm_weight >= model.weights.size()) return {};
        impl.final_norm = RmsNorm_vulkan::create(model.weights.at(model.final_norm_weight), model.descriptor.norm_epsilon,
                                                 model.descriptor.norm_weight_offset, impl.context, impl.opt);
        if (!impl.final_norm) return {};
    }
    else if (model.descriptor.final_norm != NormType::None)
        return {};
    if (!impl.lm_head) return {};
    impl.bytes = impl.attention_hyper->device_bytes();
    return result;
#else
    (void)model;
    return {};
#endif
}

#if NCNN_MOE_WITH_VULKAN
static bool qwen_ple_rows(const WeightStore& weights, const PleBlockPlan& plan,
                          std::span<const int32_t> tokens, const LayerCache& cache, ActivationBuffer& packed,
                          uint32_t& encoding, uint32_t& stride, std::vector<int32_t>& next_history)
{
    if (plan.ngram_size < 2 || plan.heads_per_ngram == 0 || plan.embedding_shards.empty()
        || cache.ple_token_history.size() > plan.ngram_size - 1) return false;
    const uint32_t heads = (plan.ngram_size - 1) * plan.heads_per_ngram;
    const uint32_t dimension = plan.embedding_dimension / heads;
    const auto& first = weights.at(plan.embedding_shards.front());
    const size_t bytes = first.dtype == DType::BFloat16  ? dimension * sizeof(uint16_t)
                         : first.dtype == DType::Float32 ? dimension * sizeof(float)
                                                         : qnk_storage_bytes(first.dtype, 1, dimension);
    if (bytes == 0 || bytes > UINT32_MAX - 3) return false;
    stride = static_cast<uint32_t>((bytes + 3) / 4 * 4);
    encoding = first.dtype == DType::IQ4NL ? 20 : first.dtype == DType::Q8_0   ? 8
                                              : first.dtype == DType::BFloat16 ? 30
                                                                               : 0;
    packed.reset(tokens.size() * heads, stride / sizeof(float), true);
    const auto multipliers = weights.at(plan.hash_multipliers).int64_values();
    const auto sizes = weights.at(plan.head_vocabulary_sizes).int64_values();
    const auto offsets = weights.at(plan.head_offsets).int64_values();
    if (multipliers.size() != plan.ngram_size || sizes.size() != heads || offsets.size() != heads) return false;
    for (size_t row = 0; row < tokens.size(); ++row)
    {
        uint64_t mixed = static_cast<uint64_t>(tokens[row]) * static_cast<uint64_t>(multipliers[0]);
        bool eos = false;
        uint32_t head = 0;
        for (uint32_t ngram = 2; ngram <= plan.ngram_size; ++ngram)
        {
            uint32_t shift = ngram - 1;
            int32_t previous = static_cast<int32_t>(plan.eos_token_id);
            if (row >= shift)
                previous = tokens[row - shift];
            else if (shift - row <= cache.ple_token_history.size())
                previous = cache.ple_token_history[cache.ple_token_history.size() - (shift - row)];
            eos = eos || previous == static_cast<int32_t>(plan.eos_token_id);
            int64_t token = eos ? static_cast<int64_t>(plan.eos_token_id) : previous;
            mixed ^= static_cast<uint64_t>(token) * static_cast<uint64_t>(multipliers[shift]);
            for (uint32_t local = 0; local < plan.heads_per_ngram; ++local, ++head)
            {
                if (sizes[head] <= 0 || offsets[head] < 0) return false;
                int64_t remainder = std::bit_cast<int64_t>(mixed) % sizes[head];
                if (remainder < 0) remainder += sizes[head];
                uint64_t index = static_cast<uint64_t>(remainder + offsets[head]);
                bool found = false;
                for (auto handle : plan.embedding_shards)
                {
                    const auto& table = weights.at(handle);
                    if (index >= table.shape[0])
                    {
                        index -= table.shape[0];
                        continue;
                    }
                    if (table.dtype != first.dtype || table.shape[1] != dimension) return false;
                    const uint8_t* source;
                    size_t source_size;
                    if (table.dtype == DType::BFloat16)
                    {
                        const auto view = table.bfloat16_values();
                        source = reinterpret_cast<const uint8_t*>(view.data());
                        source_size = view.size_bytes();
                    }
                    else if (table.dtype == DType::Float32)
                    {
                        const auto view = table.float32_values();
                        source = reinterpret_cast<const uint8_t*>(view.data());
                        source_size = view.size_bytes();
                    }
                    else
                    {
                        const auto view = table.qnk_values();
                        source = view.data();
                        source_size = view.size();
                    }
                    if (!source || index * bytes > source_size || bytes > source_size - index * bytes) return false;
                    std::memcpy(packed.mutable_row_bytes(row * heads + head).data(), source + index * bytes, bytes);
                    found = true;
                    break;
                }
                if (!found) return false;
            }
        }
    }
    const size_t length = plan.ngram_size - 1;
    next_history = cache.ple_token_history;
    if (tokens.size() >= length)
        next_history.assign(tokens.end() - length, tokens.end());
    else
    {
        next_history.insert(next_history.begin(), length - next_history.size(), static_cast<int32_t>(plan.eos_token_id));
        std::move(next_history.begin() + tokens.size(), next_history.end(), next_history.begin());
        std::copy(tokens.begin(), tokens.end(), next_history.end() - tokens.size());
    }
    return true;
}
#endif

bool QwenLayer_vulkan::forward(const ActivationBuffer& initial_hidden, std::span<const int32_t> tokens,
                               uint64_t position, LayerCache& cache, const ExpertDispatchOptions& options,
                               ExpertDispatchPlan& routes, QwenLayerWorkspace_vulkan& workspace, std::string& error) const
{
#if NCNN_MOE_WITH_VULKAN
    error.clear();
    const auto fail = [&](const char* stage) { error = stage; return false; };
    const auto& impl = *d;
    auto& previous = *workspace.d;
    if (tokens.empty() || tokens.size() > static_cast<size_t>(std::numeric_limits<int>::max())
        || cache.transaction.active || cache.vulkan_attention_state_unknown
        || (previous.context && (previous.context != impl.context || !previous.pending || previous.pending->rows() != tokens.size()))) return fail("invalid cache, token shape or device transition");
    ActivationBuffer packed;
    uint32_t encoding = 0, stride = 0;
    std::vector<int32_t> next_history;
    if (impl.ple.enabled() && ((!cache.ple_device_state && !cache.ple_convolution_state.empty()) || !qwen_ple_rows(*impl.weights, impl.ple, tokens, cache, packed, encoding, stride, next_history))) return fail("PLE row selection or incompatible prefix");
    auto transfer = impl.context->acquire_transfer_slot();
    auto& slot = transfer.slot();
    const auto previous_post = previous.post;
    const auto previous_combiner = previous.combiner;
    const uint64_t previous_dispatches = previous.context ? previous.pending_dispatches + 1 : 0;
    auto ple_state = cache.ple_device_state;
    const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
    if (slot.command_used && slot.command->reset() != 0) return fail("command buffer reset");
    slot.command_used = true;
    auto& cmd = *slot.command;
    std::vector<ncnn::VkMat> storage;
    GatedDeltaWork_vulkan delta_work;
    QsaAttentionWork_vulkan qsa_work;
    RouterWorkspace_vulkan router_work;
    uint64_t uploads = 0;
    ncnn::VkMat input, residual, attention_input, attention_inject, attention_output, normalized, ffn_inject;
    ncnn::VkMat packed_gpu, packed_staging, ple_norm, ple_key, ple_query, ple_value, gated, conv_input;
    if (previous.context)
    {
        if (!previous.record_hidden(input, cmd, storage, uploads)) return fail("previous FFN combine and residual");
    }
    else
    {
        if (initial_hidden.rows() != tokens.size() || initial_hidden.columns() != impl.hidden * impl.copies
            || initial_hidden.dtype() != DType::Float32 || !fill_staging_upload(initial_hidden, slot.upload, slot.staging_allocator)
            || !record_mapped_upload(slot.upload, input, cmd, impl.opt)) return fail("initial activation upload");
        ++uploads;
    }
    const uint64_t old_ple_bytes = ple_state ? ple_state->allocated_bytes() : 0;
    uint32_t next_slot = 0;
    if (impl.ple.enabled())
    {
        const uint32_t heads = (impl.ple.ngram_size - 1) * impl.ple.heads_per_ngram;
        const uint32_t length = (impl.ple.convolution_kernel_size - 1) * impl.ple.ngram_size;
        const uint32_t expanded = impl.hidden * impl.copies;
        if (ple_state && (!ple_state->d->valid || ple_state->d->context != impl.context)) return fail("invalid PLE device state");
        if (!ple_state)
        {
            ple_state = std::make_shared<PleState_vulkan>();
            ple_state->d->context = impl.context;
            ncnn::Mat zero(static_cast<int>(std::max(1u, length) * expanded), sizeof(float));
            zero.fill(0.0f);
            cmd.record_upload(zero, ple_state->d->convolution, impl.opt);
        }
        if (!fill_staging_upload(packed, packed_staging, slot.staging_allocator)
            || !record_mapped_upload(packed_staging, packed_gpu, cmd, impl.opt)
            || !qwen_allocate(ple_norm, impl.ple.embedding_dimension, input.h, impl.context)) return fail("PLE selected-row upload or workspace allocation");
        ++uploads;
        qwen_record(impl.pipeline.get(), 7, impl.ple.embedding_dimension, heads, input.h, impl.ple.embedding_dimension,
                    0, 0, {packed_gpu, packed_gpu, packed_gpu, packed_gpu, packed_gpu, ple_norm}, cmd,
                    encoding, impl.ple.embedding_dimension / heads, stride);
        if (!impl.ple_key->record_forward(ple_norm, ple_key, cmd, storage)
            || !impl.ple_value->record_forward(ple_norm, ple_value, cmd, storage)) return fail("PLE key/value projection");
        ncnn::VkMat key_normalized, query_normalized;
        if (!qwen_allocate(key_normalized, expanded, input.h, impl.context)
            || !qwen_allocate(query_normalized, expanded, input.h, impl.context)
            || !qwen_allocate(gated, expanded, input.h, impl.context)
            || !qwen_allocate(conv_input, expanded, input.h, impl.context)
            || !qwen_allocate(ple_query, expanded, input.h, impl.context)) return fail("PLE workspace allocation");
        qwen_prepare_read(impl.key_norm);
        qwen_prepare_read(impl.query_norm);
        qwen_prepare_read(impl.conv_norm);
        qwen_prepare_read(impl.convolution_weight);
        qwen_prepare_read(ple_state->d->convolution);
        qwen_record(impl.pipeline.get(), 0, impl.hidden, impl.copies, input.h, expanded, impl.epsilon, impl.offset,
                    {ple_key, impl.key_norm, ple_key, ple_key, ple_key, key_normalized}, cmd);
        qwen_record(impl.pipeline.get(), 0, impl.hidden, impl.copies, input.h, expanded, impl.epsilon, impl.offset,
                    {input, impl.query_norm, input, input, input, query_normalized}, cmd);
        qwen_record(impl.pipeline.get(), 5, impl.hidden, impl.copies, input.h, expanded, 0, 0,
                    {key_normalized, query_normalized, ple_value, ple_value, ple_value, gated}, cmd);
        qwen_record(impl.pipeline.get(), 0, impl.hidden, impl.copies, input.h, expanded, impl.epsilon, impl.offset,
                    {gated, impl.conv_norm, gated, gated, gated, conv_input}, cmd);
        qwen_record(impl.pipeline.get(), 6, impl.hidden, impl.copies, input.h, expanded, 0, 0,
                    {conv_input, gated, impl.convolution_weight, input, ple_state->d->convolution, ple_query}, cmd,
                    impl.ple.convolution_kernel_size, impl.ple.ngram_size, length, ple_state->d->slot);
        next_slot = length == 0 ? 0 : static_cast<uint32_t>((ple_state->d->slot + tokens.size()) % length);
        storage.insert(storage.end(), {input, key_normalized, query_normalized});
        input = ple_query;
    }
    if (impl.qsa && previous.context)
    {
        // Keep the combined residual on the device while crossing an explicit
        // compute-to-transfer-to-compute dependency boundary before QSA.
        // Retain both allocations through this command's fence wait.
        ncnn::VkMat independent_input;
        cmd.record_clone(input, independent_input, impl.opt);
        if (independent_input.empty()) return fail("QSA device input copy");
        storage.push_back(input);
        input = std::move(independent_input);
    }
    if (!impl.attention_hyper->record_pre(input, attention_input, attention_inject, cmd, storage)) return fail("attention gated residual pre");
    if (impl.delta)
    {
        if (!impl.delta->record_forward(attention_input, cache, tokens.size(), attention_output, cmd, delta_work)) return fail("Gated DeltaNet recording");
    }
    else if (!impl.qsa->record_forward(attention_input, cache, position, attention_output, cmd, qsa_work))
        return fail("QSA recording");
    if (!impl.attention_hyper->record_post(attention_output, input, attention_inject, residual, cmd)
        || !impl.ffn_hyper->record_pre(residual, normalized, ffn_inject, cmd, storage)
        || !impl.router->record(normalized, options, router_work, cmd)) return fail("attention post, FFN pre or GPU router");
    const uint32_t selected_columns = Router_vulkan::selected_columns(options.top_k);
    if (!prepare_staging_batch(slot.download, tokens.size(), selected_columns, slot.staging_allocator)
        || !record_prepared_activation_staging_download(router_work.selected, tokens.size(), selected_columns,
                                                        slot.download, cmd, impl.context->device(), impl.router->option())) return fail("route metadata staging");
    if (submit_compute_and_wait(cmd, impl.context->device()) != 0)
    {
        if (impl.delta) delta_work.mark_submit_failed();
        if (ple_state) ple_state->d->valid = false;
        cache.vulkan_attention_state_unknown = true;
        return fail("Vulkan command submission");
    }
    struct PublicationGuard
    {
        LayerCache& cache;
        bool* ple_valid;
        bool committed = false;
        ~PublicationGuard()
        {
            if (!committed)
            {
                cache.vulkan_attention_state_unknown = true;
                if (ple_valid) *ple_valid = false;
            }
        }
    } publication{cache, ple_state ? &ple_state->d->valid : nullptr};
    previous.selected.reset(tokens.size(), selected_columns, false);
    auto device_input = std::make_shared<DeviceTensor_vulkan>();
    if (!copy_staging_to_cpu_batch(slot.download, previous.selected)) return fail("route metadata download");
    auto decoded = Router_vulkan::decode_selected(previous.selected, options, routes);
    if (!decoded)
    {
        error = "GPU route selection: " + decoded.error().message;
        return false;
    }
    if (!device_input->assign_completed(normalized, impl.context)) return fail("normalized device publication");
    if (impl.delta ? !delta_work.commit() : !qsa_work.commit()) return fail("attention cache commit");
    if (impl.ple.enabled())
    {
        ple_state->d->slot = next_slot;
        cache.ple_device_state = ple_state;
        cache.ple_token_history = std::move(next_history);
        cache.ple_first_slot = next_slot;
        cache.device_allocated_size += ple_state->allocated_bytes() - old_ple_bytes;
    }
    previous.clear_buffers();
    previous.context = impl.context;
    previous.post = impl.ffn_hyper;
    previous.combiner = impl.combiner;
    previous.normalized = std::move(device_input);
    previous.residual = std::move(residual);
    previous.injection = std::move(ffn_inject);
    publication.committed = true;
    auto& counters = impl.context->runtime_state();
    ++counters.compute_submissions;
    counters.batch_uploads += uploads;
    ++counters.batch_downloads;
    counters.dispatches += previous_dispatches + (impl.ple.enabled() ? 25 : 17);
    return true;
#else
    (void)initial_hidden;
    (void)tokens;
    (void)position;
    (void)cache;
    (void)options;
    (void)routes;
    (void)workspace;
    error = "Vulkan is unavailable";
    return false;
#endif
}

bool QwenLayer_vulkan::defer_combine(std::span<const MoeCombineInput_vulkan> inputs,
                                     size_t rows, uint32_t columns, std::shared_ptr<const DeviceTensor_vulkan> shared, QwenLayerWorkspace_vulkan& workspace)
{
#if NCNN_MOE_WITH_VULKAN
    if (!workspace.active()) return false;
    for (const auto& input : inputs)
        if (!input.device || input.device->empty()) return false;
    const bool has_shared = static_cast<bool>(shared);
    auto pending = workspace.d->combiner->prepare(inputs, rows, columns, nullptr, std::move(shared));
    if (!pending) return false;
    const std::lock_guard<std::mutex> lock(workspace.d->context->command_mutex());
    workspace.d->pending = std::move(pending);
    workspace.d->pending_dispatches = inputs.size() + (has_shared ? 1 : 0);
    return true;
#else
    (void)inputs;
    (void)rows;
    (void)columns;
    (void)shared;
    (void)workspace;
    return false;
#endif
}

bool QwenLayer_vulkan::finish(const CompiledModel& model, bool produce_logits, bool last_only,
                              ActivationBuffer& logits, QwenLayerWorkspace_vulkan& workspace)
{
#if NCNN_MOE_WITH_VULKAN
    if (!workspace.active() || !model.qwen_device_head) return false;
    if (!produce_logits) return true;
    auto& head = *model.qwen_device_head->d;
    if (head.context != workspace.d->context) return false;
    auto transfer = head.context->acquire_transfer_slot();
    auto& slot = transfer.slot();
    const std::lock_guard<std::mutex> lock(head.context->command_mutex());
    if (slot.command_used && slot.command->reset() != 0) return false;
    slot.command_used = true;
    auto& cmd = *slot.command;
    std::vector<ncnn::VkMat> storage;
    ncnn::VkMat hidden, reduced, unused_injection, normalized, output;
    uint64_t uploads = 0;
    if (!workspace.d->record_hidden(hidden, cmd, storage, uploads)) return false;
    storage.push_back(hidden);
    if (last_only) hidden = row_view(hidden, static_cast<size_t>(hidden.h - 1), 1);
    if (!head.attention_hyper->record_pre(hidden, reduced, unused_injection, cmd, storage)) return false;
    if (head.final_norm)
    {
        if (!head.final_norm->record(reduced, normalized, cmd)) return false;
    }
    else
        normalized = reduced;
    if (!head.lm_head->record_forward(normalized, output, cmd, storage)) return false;
    if (!prepare_staging_batch(slot.download, output.h, output.w, slot.staging_allocator)
        || !record_prepared_activation_staging_download(output, output.h, output.w, slot.download, cmd, head.context->device(), head.opt)
        || submit_compute_and_wait(cmd, head.context->device()) != 0) return false;
    logits.reset(output.h, output.w, false);
    if (!copy_staging_to_cpu_batch(slot.download, logits)) return false;
    auto& counters = head.context->runtime_state();
    ++counters.compute_submissions;
    counters.batch_uploads += uploads;
    ++counters.batch_downloads;
    counters.dispatches += workspace.d->pending_dispatches + 1 + 6 + (head.final_norm ? 1 : 0);
    return true;
#else
    (void)model;
    (void)produce_logits;
    (void)last_only;
    (void)logits;
    (void)workspace;
    return false;
#endif
}
} // namespace moe
} // namespace ncnn
