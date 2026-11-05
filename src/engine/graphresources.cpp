#include "graphresources.h"

#include "backends/ncnn/modelpipeline.h"
#include "backends/ncnn/gpusegment_vulkan.h"
#include "backends/ncnn/expertbank_vulkan.h"
#include "backends/ncnn/vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "graph/compiledmodel.h"
#include "storage/expertcache.h"

#if NCNN_MOE_WITH_VULKAN
#include <gpu.h>
#endif

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace ncnn {
namespace moe {

static std::map<uint32_t, GpuResourceRequest> graph_resource_requests(const ModelGraphMemoryPlan& graph, uint32_t sessions)
{
    std::map<uint32_t, GpuResourceRequest> requests;
    for (const GraphMemoryDevice& device : graph.prefill.devices)
        requests[device.device_index].workspace_size = device.peak_size;
    for (const GraphMemoryDevice& device : graph.decode.devices)
    {
        // Decode already describes the complete multi-sequence graph. Divide
        // its peak before applying the per-Session ledger multiplier below.
        const uint64_t per_session = device.peak_size / sessions + (device.peak_size % sessions != 0);
        requests[device.device_index].workspace_size = std::max(requests[device.device_index].workspace_size, per_session);
    }
    for (const GraphMemoryDevice& device : graph.session_state)
        requests[device.device_index].session_state_size = device.peak_size;
    return requests;
}

static Result<void> allocate_graph_resources(const CompiledModel& model,
                                             const std::map<uint32_t, GpuResourceRequest>& requests,
                                             uint32_t sessions,
                                             std::vector<GpuResourcePlan>* initial_resources)
{
    for (const auto& [device, request] : requests)
    {
        if (request.session_state_size > std::numeric_limits<uint64_t>::max() - request.workspace_size
            || request.session_state_size + request.workspace_size > std::numeric_limits<uint64_t>::max() / sessions)
            return Error{ErrorCode::InvalidArgument, "GPU graph memory size overflows"};
        const uint64_t size = (request.session_state_size + request.workspace_size) * sessions;
        if (size == 0)
            continue;
        auto allocated = reserve_vulkan_graph_compute(model.vulkan_runtime, device, model.opt.optimization_flags, size);
        if (!allocated)
            return Error{allocated.error().code, "GPU graph allocation failed on device " + std::to_string(device) + ": " + allocated.error().message};
        if (!initial_resources)
            continue;
        auto resource = std::find_if(initial_resources->begin(), initial_resources->end(), [device](const GpuResourcePlan& plan) {
            return plan.initial.device_index == device;
        });
        if (resource == initial_resources->end())
            return Error{ErrorCode::InternalError, "GPU graph device is missing from the resource plan"};
        resource->admitted.session_state_size = request.session_state_size;
        resource->admitted.workspace_size = request.workspace_size;
        // A successful reservation covers the complete execution capacity,
        // including any free preparation backing reused by the arena.
        const VulkanGraphArenaSize& physical = allocated.value();
        const uint64_t compute_physical_size = physical.total_physical_size - physical.preparation_physical_size;
        resource->allocated_execution_size = std::max(size, compute_physical_size);
    }
    return {};
}

static Result<void> prepare_model_graph_resources(CompiledModel& model,
                                                  uint32_t prefill_rows,
                                                  ModelGraphMemoryPlan&& plan)
{
    const uint32_t sessions = std::max(1u, model.opt.num_concurrent_sessions);
    if (prefill_rows == 0 || plan.prefill_rows != prefill_rows || plan.sequence_count != sessions)
        return Error{ErrorCode::InvalidArgument, "GPU graph memory plan does not match the requested shape"};
    auto resources = std::make_shared<GraphResources>();
    const auto requests = graph_resource_requests(plan, sessions);
    for (const auto& [device, request] : requests)
    {
        if (request.workspace_size == 0 && request.session_state_size == 0)
            continue;
        auto context = retain_vulkan_graph_context(model.vulkan_runtime, device, model.opt.optimization_flags);
        if (!context)
            return Error{ErrorCode::InternalError, "cannot create the planned GPU graph resource domain"};
        resources->contexts.push_back(std::move(context));
    }
    auto allocated = allocate_graph_resources(model, requests, sessions, &model.memory_plan.gpu_devices);
    if (!allocated)
        return allocated.error();
    resources->plan = std::move(plan);
    model.graph_resources = std::move(resources);
    return {};
}

Result<void> prepare_model_graph_resources(CompiledModel& model, uint32_t prefill_rows)
{
    const uint32_t rows = std::max(1u, prefill_rows);
    const uint32_t sessions = std::max(1u, model.opt.num_concurrent_sessions);
    auto plan = plan_model_graph_memory(model, rows, sessions);
    if (!plan)
        return plan.error();
    return prepare_model_graph_resources(model, rows, std::move(plan).value());
}

#if NCNN_MOE_WITH_VULKAN
static bool bank_chain_supported(const CompiledModel& model)
{
    const size_t layers = model.graph.layer_plans.size();
    if (model.opt.hybrid_mode != HybridMode::HybridExperts || !model.gpu_head || layers == 0
        || model.gpu_segments.size() != layers || model.speculative.enabled()
        || !model.speculative.graph.nodes.empty() || has_flag(model.opt.flags, OptionRouterPrediction)
        || has_flag(model.opt.flags, OptionDisableGpuExpertExecution)) return false;
    const auto context = model.gpu_head ? model.gpu_head->context() : nullptr;
    if (!context || std::any_of(model.gpu_segments.begin(), model.gpu_segments.end(), [&](const auto& segment) {
            return !segment || segment->context() != context;
        })) return false;
    // The bank executor bypasses exactly this scheduled chain. Any additional
    // host consumer or a different ordering keeps explicit GPU streaming.
    size_t position = 0;
    const auto next = [&](ExecutionNodeType type, size_t layer, bool gpu) {
        if (position >= model.schedule.node_order.size()) return false;
        const ExecutionNode& node = model.graph.nodes[model.schedule.node_order[position]];
        const bool ok = node.type == type && (layer >= layers || node.layer_plan_index == layer)
                        && (!gpu || node.backend == ExecutionBackend::Vulkan);
        if (!ok) return false;
        ++position;
        return true;
    };
    if (!next(ExecutionNodeType::TokenEmbedding, layers, false)) return false;
    for (size_t layer = 0; layer < layers; ++layer)
    {
        const CompiledLayerPlan& plan = model.graph.layer_plans[layer];
        if (plan.dense_ffn.down_weight != invalid_tensor_handle)
        {
            if (!next(ExecutionNodeType::Attention, layer, true)
                || !next(ExecutionNodeType::DenseFfn, layer, true)) return false;
            continue;
        }
        const MoeBlockPlan& moe = plan.moe;
        if (moe.experts.empty() || moe.top_k == 0 || moe.top_k > 64 || moe.top_k > moe.experts.size()
            || !next(ExecutionNodeType::Attention, layer, true)
            || !next(ExecutionNodeType::Router, layer, true)
            || !next(ExecutionNodeType::ExpertDispatch, layer, false)
            || !next(ExecutionNodeType::ExpertGroup, layer, true)
            || (moe.has_shared_expert && !next(ExecutionNodeType::SharedExpertGroup, layer, true))
            || !next(ExecutionNodeType::Combine, layer, true)) return false;
    }
    return next(ExecutionNodeType::FinalNorm, layers, true) && next(ExecutionNodeType::LmHead, layers, true)
           && position == model.schedule.node_order.size();
}

static ExpertBankWeights_vulkan bank_expert_weights(const CompiledModel& model, const ExpertPlan& expert)
{
    const auto tensor = [&](TensorHandle handle) -> const TensorData* {
        return handle == invalid_tensor_handle ? nullptr : &model.weights.at(handle);
    };
    ExpertBankWeights_vulkan result;
    result.gate_up = tensor(expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight : expert.gate_weight);
    result.up = tensor(expert.up_weight);
    result.down = tensor(expert.down_weight);
    result.gate_up_bias = tensor(expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_bias : expert.gate_bias);
    result.up_bias = tensor(expert.up_bias);
    result.down_bias = tensor(expert.down_bias);
    result.layout = expert.layout;
    result.activation = expert.activation;
    result.activation_limit = expert.activation_limit;
    return result;
}

static bool deferred_bank_weight(const TensorData* tensor)
{
    return tensor && tensor->dtype == DType::MxFp4
           && (tensor->mxfp4_blocks.size() != tensor->element_count() / 2
               || tensor->mxfp4_scales.size() != tensor->element_count() / 32);
}

static Result<ExpertBankWeights_vulkan> load_bank_expert(const CompiledModel& model, const ExpertPlan& expert, ExpertCache& cache)
{
    ExpertBankWeights_vulkan result = bank_expert_weights(model, expert);
    if (!deferred_bank_weight(result.gate_up) && !deferred_bank_weight(result.up) && !deferred_bank_weight(result.down)) return result;
    struct Leases
    {
        ExpertCacheLease gate;
        ExpertCacheLease up;
    };
    auto owner = std::make_shared<Leases>();
    auto gate = cache.acquire_pair(*result.gate_up, *result.down);
    if (!gate) return gate.error();
    owner->gate = std::move(gate).value();
    result.gate_up = owner->gate.gate_up.get();
    result.down = owner->gate.down.get();
    if (result.up)
    {
        auto up = cache.acquire_pair(*result.up, model.weights.at(expert.down_weight));
        if (!up) return up.error();
        owner->up = std::move(up).value();
        result.up = owner->up.gate_up.get();
    }
    result.owner = std::move(owner);
    return result;
}

static GpuResourceSnapshot bank_live_resources(const CompiledModel& model, uint32_t device)
{
    GpuResourceSnapshot result{device, 0, 0};
    const auto context = VulkanContext::find(device, model.vulkan_runtime, model.opt.optimization_flags);
    if (context)
    {
        result.heap_budget = static_cast<uint64_t>(context->device()->get_heap_budget()) * 1024 * 1024;
        result.heap_usage = static_cast<uint64_t>(context->device()->get_heap_usage()) * 1024 * 1024;
    }
    return result;
}
#endif

Result<void> prepare_model_expert_banks(CompiledModel& model, uint32_t prefill_rows, uint64_t expert_budget)
{
    if (!model.gpu_expert_banks.empty())
        return Error{ErrorCode::InvalidArgument, "resident Expert banks have already been prepared"};
#if NCNN_MOE_WITH_VULKAN
    auto valid = model.schedule.validate(model.graph);
    if (!valid) return valid.error();
    if (bank_chain_supported(model))
    {
        const auto context = model.gpu_head->context();
        const uint32_t device = model.graph.layer_plans.front().vulkan_device_index;
        uint64_t total = 0, lease_capacity = 1;
        std::vector<ExpertBankPlan_vulkan> plans;
        bool supported = true;
        for (const CompiledLayerPlan& layer : model.graph.layer_plans)
        {
            if (layer.dense_ffn.down_weight != invalid_tensor_handle)
            {
                plans.emplace_back();
                continue;
            }
            std::vector<ExpertBankWeights_vulkan> experts;
            for (const ExpertPlan& expert : layer.moe.experts)
            {
                experts.push_back(bank_expert_weights(model, expert));
                const auto& weights = experts.back();
                if ((deferred_bank_weight(weights.gate_up) || deferred_bank_weight(weights.up) || deferred_bank_weight(weights.down))
                    && (!weights.gate_up || !weights.down || !weights.gate_up->mxfp4_file_storage || !weights.down->mxfp4_file_storage
                        || (weights.up && !weights.up->mxfp4_file_storage))) supported = false;
                uint64_t elements = 0;
                for (const TensorData* matrix : {weights.gate_up, weights.up, weights.down})
                    if (matrix)
                    {
                        if (matrix->element_count() > (UINT64_MAX - elements))
                            return Error{ErrorCode::InvalidModel, "resident Expert loading size overflows"};
                        elements += matrix->element_count();
                    }
                if (elements > UINT64_MAX / (2 * sizeof(float)))
                    return Error{ErrorCode::InvalidModel, "resident Expert loading size overflows"};
                // Sequential loading retains at most separate gate/down and
                // up/down leases. This is a capacity ceiling, not an eager
                // allocation or a copy of all model Expert weights.
                lease_capacity = std::max(lease_capacity, elements * (2 * sizeof(float)));
            }
            auto plan = ExpertBank_vulkan::plan(experts, context);
            if (!plan)
            {
                if (plan.error().code != ErrorCode::UnsupportedModel) return plan.error();
                supported = false;
                break;
            }
            if (plan.value().input_columns != model.descriptor.hidden_size || plan.value().output_columns != model.descriptor.hidden_size)
            {
                supported = false;
                break;
            }
            if (plan.value().physical_size > UINT64_MAX - total)
                return Error{ErrorCode::InvalidModel, "resident Expert bank physical size overflows"};
            total += plan.value().physical_size;
            plans.emplace_back(std::move(plan).value());
        }
        if (supported && (expert_budget == 0 || total <= expert_budget))
        {
            // Keep the full plan metadata visible to the resident memory plan;
            // the selected prefix is truncated below.
            model.gpu_expert_bank_plans = plans;
            model.resident_submission_rows = std::max(1u, prefill_rows);
            auto graph = plan_model_graph_memory(model, model.resident_submission_rows, std::max(1u, model.opt.num_concurrent_sessions));
            if (!graph) return graph.error();
            auto candidate = model.memory_plan.gpu_devices;
            auto resource = std::find_if(candidate.begin(), candidate.end(), [&](const auto& item) { return item.initial.device_index == device; });
            const auto requests = graph_resource_requests(graph.value(), std::max(1u, model.opt.num_concurrent_sessions));
            const auto request = requests.find(device);
            bool fits = resource != candidate.end() && request != requests.end();
            uint64_t bank_total = total;
            size_t bank_layers = plans.size();
            if (fits)
            {
                resource->expert_cache_reserve = 0;
                resource->admitted.session_state_size = request->second.session_state_size;
                resource->admitted.workspace_size = request->second.workspace_size;
                // Baseline estimates and actual free heap are independent.
                // The former also supports direct compiler byte allowances;
                // actual usage is never compared with an artificial budget.
                const GpuResourceSnapshot baseline{device, resource->initial.heap_budget, resource->initial.heap_usage};
                const auto live = bank_live_resources(model, device);
                const uint64_t execution = std::max(gpu_resource_execution_reserve(*resource), resource->allocated_execution_size);
                const uint64_t pending = execution - resource->allocated_execution_size;
                // Bank admission leaves headroom for the attention/FFN groups
                // that are committed after the bank. A partial chain also keeps
                // room for the streaming cache used by its tail layers.
                constexpr uint64_t bank_headroom = UINT64_C(256) * 1024 * 1024;
                constexpr uint64_t streaming_reserve = UINT64_C(1024) * 1024 * 1024;
                const bool free_heap = live.heap_usage <= live.heap_budget
                                       && pending <= live.heap_budget - live.heap_usage;
                const uint64_t available = free_heap ? live.heap_budget - live.heap_usage - pending : 0;
                uint64_t maximum = free_heap && available > bank_headroom ? available - bank_headroom : 0;
                if (expert_budget != 0)
                    maximum = std::min(maximum, expert_budget);
                const auto fit_prefix = [&](uint64_t budget, uint64_t& bytes, size_t& count) {
                    bytes = 0;
                    count = 0;
                    for (size_t index = 0; index < plans.size(); ++index)
                    {
                        const uint64_t size = plans[index].physical_size;
                        if (size != 0 && (size > budget || bytes > budget - size))
                            break;
                        bytes += size;
                        count = index + 1;
                    }
                };
                uint64_t prefix = 0;
                size_t prefix_layers = 0;
                fit_prefix(maximum, prefix, prefix_layers);
                if (prefix_layers != plans.size())
                {
                    maximum = maximum > streaming_reserve ? maximum - streaming_reserve : 0;
                    fit_prefix(maximum, prefix, prefix_layers);
                    // The ordinary tail path owns dense FFNs. A bank can only
                    // cover a prefix, so keep partial banks dense-free after it.
                    for (size_t index = prefix_layers; index < plans.size(); ++index)
                        if (model.graph.layer_plans[index].dense_ffn.down_weight != invalid_tensor_handle)
                        {
                            prefix = 0;
                            prefix_layers = 0;
                            break;
                        }
                }
                if (prefix_layers != plans.size())
                {
                    for (size_t index = prefix_layers; index < plans.size(); ++index)
                        plans[index] = ExpertBankPlan_vulkan{};
                    bank_total = prefix;
                    bank_layers = prefix_layers;
                }
                fits = prefix_layers != 0 && free_heap && available > bank_headroom
                       && bank_total <= maximum
                       && can_admit_gpu_resources(*resource, baseline, {bank_total, 0, 0});
                if (fits)
                    model.resident_bank_layers = static_cast<uint32_t>(bank_layers);
            }
            if (fits)
            {
                // Selection ends here. All subsequent failures are terminal;
                // no partially prepared bank is changed to streaming/CPU.
                model.memory_plan.gpu_devices = std::move(candidate);
                model.gpu_expert_bank_plans = plans;
                auto prepared = prepare_model_graph_resources(model, model.resident_submission_rows, std::move(graph).value());
                if (!prepared) return prepared.error();
                const auto live = bank_live_resources(model, device);
                if (live.heap_usage > live.heap_budget || bank_total > live.heap_budget - live.heap_usage)
                    return Error{ErrorCode::InvalidArgument, "selected resident Expert banks exceed the live GPU budget after graph reservation"};
                ExpertCache cache(lease_capacity, 1, {}, model.opt.expert_io_mode);
                std::vector<std::shared_ptr<ExpertBank_vulkan>> banks;
                uint64_t physical = 0;
                for (size_t index = 0; index < model.graph.layer_plans.size(); ++index)
                {
                    const auto& layer = model.graph.layer_plans[index];
                    if (index >= bank_layers || layer.dense_ffn.down_weight != invalid_tensor_handle)
                    {
                        banks.push_back({});
                        continue;
                    }
                    auto bank = ExpertBank_vulkan::create(model.gpu_expert_bank_plans[index], context, [&](uint32_t expert) {
                        return load_bank_expert(model, layer.moe.experts.at(expert), cache);
                    });
                    if (!bank)
                        return Error{bank.error().code, "selected resident Expert bank layer " + std::to_string(index) + " on device "
                                                            + std::to_string(device) + ": " + bank.error().message};
                    if (bank.value()->physical_size() > UINT64_MAX - physical)
                        return Error{ErrorCode::InvalidModel, "resident Expert bank physical size overflows"};
                    physical += bank.value()->physical_size();
                    banks.push_back(std::move(bank).value());
                }
                if (physical != bank_total)
                    return Error{ErrorCode::InternalError, "selected resident Expert bank allocation does not match its admitted size"};
                for (auto& item : model.memory_plan.gpu_devices)
                    if (item.initial.device_index == device) commit_gpu_resources(item, {physical, 0, 0});
                model.gpu_expert_bank_plans = std::move(plans);
                model.gpu_expert_banks = std::move(banks);
                model.gpu_expert_bank_size = physical;
                model.expert_backend.reset();
                model.expert_cache.reset();
                model.opt.expert_gpu_cache_size = model.opt.expert_gpu_victim_cache_size = 0;
                return {};
            }
            model.gpu_expert_bank_plans.clear();
            model.gpu_expert_banks.clear();
            model.resident_bank_layers = 0;
            model.resident_submission_rows = 0;
        }
    }
#else
    (void)expert_budget;
#endif
    return prepare_model_graph_resources(model, prefill_rows);
}

Result<uint64_t> ensure_model_graph_resources(const CompiledModel& model, uint64_t rows, uint64_t context_tokens, uint32_t sequence_count)
{
    // Direct compiler clients may manage allocation themselves. Runtime loads
    // always initialize the shared graph resource domain before publishing.
    if (!model.graph_resources)
        return uint64_t{0};
    GraphResources& resources = *model.graph_resources;
    if (model.memory_plan.gpu_devices.empty())
    {
        const std::lock_guard<std::mutex> lock(resources.mutex);
        return resources.plan.host_output_slot_size;
    }
    if (rows > std::numeric_limits<uint32_t>::max() || context_tokens > std::numeric_limits<uint32_t>::max())
        return Error{ErrorCode::InvalidArgument, "GPU graph shape exceeds uint32 range"};
    // The row count may decrease when context grows. Read the shape under one
    // lock so concurrent Sessions cannot combine dimensions from two plans.
    const std::lock_guard<std::mutex> lock(resources.mutex);
    if (rows <= resources.plan.prefill_rows && context_tokens <= resources.plan.context_tokens
        && sequence_count <= resources.plan.sequence_count)
        return resources.plan.host_output_slot_size;
    // Replan the requested shape, not the Cartesian product of all historic
    // maxima. A long-context decode can reuse a larger short-context prefill
    // arena without reserving a fictitious long-context 512-row prefill.
    const uint32_t next_rows = std::max(1u, static_cast<uint32_t>(rows));
    uint32_t next_context = std::max(1u, resources.plan.context_tokens);
    while (next_context < context_tokens)
    {
        if (next_context > std::numeric_limits<uint32_t>::max() / 2)
        {
            next_context = static_cast<uint32_t>(context_tokens);
            break;
        }
        next_context *= 2;
    }
    auto plan = plan_model_graph_memory(model, next_rows, std::max(sequence_count, resources.plan.sequence_count), next_context);
    if (!plan)
        return plan.error();
    // Published host outputs may outlive a smaller-row context replan. Keep
    // each workspace's already-admitted slot capacity until model teardown.
    plan.value().host_output_slot_size = std::max(plan.value().host_output_slot_size, resources.plan.host_output_slot_size);
    plan.value().host_output_workspace_size = std::max(plan.value().host_output_workspace_size, resources.plan.host_output_workspace_size);
    const uint32_t sessions = std::max({1u, model.opt.num_concurrent_sessions, plan.value().sequence_count});
    const auto requests = graph_resource_requests(plan.value(), sessions);
    auto allocated = allocate_graph_resources(model, requests, sessions, nullptr);
    if (!allocated)
    {
        // Replanning may have released obsolete free arena backing. Keep the
        // host-output capacity, but force the next request to reserve again.
        resources.plan.prefill_rows = 0;
        return allocated.error();
    }
    resources.plan = std::move(plan).value();
    return resources.plan.host_output_slot_size;
}

} // namespace moe
} // namespace ncnn
