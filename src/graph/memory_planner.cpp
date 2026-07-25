#include "memory_planner.h"

#include "ncnn/moe/runtime.h"

#include <algorithm>
#include <limits>
#include <string>

namespace ncnn {
namespace moe {

static constexpr uint64_t gibibyte = 1024ull * 1024ull * 1024ull;

static Result<uint64_t> checked_add(
    uint64_t left,
    uint64_t right,
    const char* name)
{
    if (right > std::numeric_limits<uint64_t>::max() - left)
        return Error{ErrorCode::InvalidModel, std::string(name) + " byte estimate overflows"};
    return left + right;
}

static Result<uint64_t> checked_multiply(
    uint64_t left,
    uint64_t right,
    const char* name)
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
        return Error{ErrorCode::InvalidModel, std::string(name) + " byte estimate overflows"};
    return left * right;
}

static Result<uint64_t> dense_bytes(const MoeIR& ir)
{
    const uint64_t element_bytes
        = ir.activation_dtype == DType::BFloat16 ? 2 : 4;
    auto embedding_elements = checked_multiply(
        ir.vocabulary_size,
        ir.hidden_size,
        "embedding");
    if (!embedding_elements)
        return embedding_elements.error();
    auto embedding_bytes = checked_multiply(
        embedding_elements.value(),
        element_bytes * 2,
        "embedding and LM head");
    if (!embedding_bytes)
        return embedding_bytes.error();
    uint64_t total = embedding_bytes.value();

    for (const LayerDescriptor& layer : ir.layers) {
        uint64_t layer_elements = ir.hidden_size;

        if (has_flag(layer.flags, LayerDescriptorAttention)) {
            auto with_attention_norm = checked_add(
                layer_elements,
                ir.hidden_size,
                "layer norms");
            if (!with_attention_norm)
                return with_attention_norm.error();
            layer_elements = with_attention_norm.value();
            const uint64_t query_size
                = static_cast<uint64_t>(layer.attention.head_count)
                  * layer.attention.head_dimension;
            const uint64_t key_value_size
                = static_cast<uint64_t>(layer.attention.kv_head_count)
                  * layer.attention.head_dimension;
            auto query = checked_multiply(
                query_size,
                ir.hidden_size,
                "query projection");
            auto key_value = checked_multiply(
                key_value_size,
                ir.hidden_size * 2ull,
                "key/value projections");
            auto output = checked_multiply(
                ir.hidden_size,
                query_size,
                "attention output projection");
            if (!query)
                return query.error();
            if (!key_value)
                return key_value.error();
            if (!output)
                return output.error();
            auto projections = checked_add(
                query.value(),
                key_value.value(),
                "attention projections");
            if (!projections)
                return projections.error();
            projections = checked_add(
                projections.value(),
                output.value(),
                "attention projections");
            if (!projections)
                return projections.error();
            auto with_projections = checked_add(
                layer_elements,
                projections.value(),
                "layer dense weights");
            if (!with_projections)
                return with_projections.error();
            layer_elements = with_projections.value();
            if (has_flag(layer.attention.flags, AttentionDescriptorBias)) {
                const uint64_t bias_elements
                    = query_size + key_value_size * 2 + ir.hidden_size;
                auto with_bias = checked_add(
                    layer_elements,
                    bias_elements,
                    "attention biases");
                if (!with_bias)
                    return with_bias.error();
                layer_elements = with_bias.value();
            }
            if (has_flag(layer.attention.flags, AttentionDescriptorSinks)) {
                auto with_sinks = checked_add(
                    layer_elements,
                    layer.attention.head_count,
                    "attention sinks");
                if (!with_sinks)
                    return with_sinks.error();
                layer_elements = with_sinks.value();
            }
        }

        auto router = checked_multiply(
            layer.ffn.moe.expert_count,
            ir.hidden_size,
            "router");
        if (!router)
            return router.error();
        auto with_router = checked_add(
            layer_elements,
            router.value(),
            "layer dense weights");
        if (!with_router)
            return with_router.error();
        layer_elements = with_router.value();
        if (has_flag(layer.ffn.moe.flags, MoeDescriptorRouterBias)) {
            auto with_router_bias = checked_add(
                layer_elements,
                layer.ffn.moe.expert_count,
                "router bias");
            if (!with_router_bias)
                return with_router_bias.error();
            layer_elements = with_router_bias.value();
        }
        if (has_flag(layer.ffn.moe.flags, MoeDescriptorProjectionBias)) {
            const uint64_t per_expert_bias
                = static_cast<uint64_t>(layer.ffn.moe.intermediate_size) * 2
                  + ir.hidden_size;
            auto expert_biases = checked_multiply(
                layer.ffn.moe.expert_count,
                per_expert_bias,
                "expert biases");
            if (!expert_biases)
                return expert_biases.error();
            auto with_expert_biases = checked_add(
                layer_elements,
                expert_biases.value(),
                "layer dense weights");
            if (!with_expert_biases)
                return with_expert_biases.error();
            layer_elements = with_expert_biases.value();
        }

        auto layer_bytes = checked_multiply(
            layer_elements,
            element_bytes,
            "layer dense weights");
        if (!layer_bytes)
            return layer_bytes.error();
        auto with_layer = checked_add(
            total,
            layer_bytes.value(),
            "dense weights");
        if (!with_layer)
            return with_layer.error();
        total = with_layer.value();
    }

    auto final_norm = checked_multiply(
        ir.hidden_size,
        element_bytes,
        "final norm");
    if (!final_norm)
        return final_norm.error();
    return checked_add(total, final_norm.value(), "dense weights");
}

static Result<uint64_t> mxfp4_pair_bytes(const MoeIR& ir)
{
    auto elements = checked_multiply(
        ir.hidden_size,
        ir.intermediate_size,
        "expert pair");
    if (!elements)
        return elements.error();
    elements = checked_multiply(
        elements.value(),
        3,
        "expert pair");
    if (!elements)
        return elements.error();
    auto encoded = checked_multiply(
        elements.value(),
        17,
        "MXFP4 expert pair");
    if (!encoded)
        return encoded.error();
    return encoded.value() / 32;
}

Result<ModelMemoryPlan> plan_model_memory(
    const MoeIR& ir,
    const RuntimeOptions& options,
    uint64_t physical_memory_bytes)
{
    ModelMemoryPlan plan;
    plan.requested_mode = options.expert_memory_mode;
    plan.physical_memory_bytes = physical_memory_bytes;
    if (options.host_memory_budget_bytes != 0) {
        if (physical_memory_bytes != 0
            && options.host_memory_budget_bytes > physical_memory_bytes) {
            return Error{
                ErrorCode::InvalidArgument,
                "host memory budget exceeds detected physical memory"};
        }
        plan.host_memory_budget_bytes = options.host_memory_budget_bytes;
    }
    else if (physical_memory_bytes != 0) {
        plan.host_memory_budget_bytes = physical_memory_bytes / 4 * 3;
    }
    else {
        plan.host_memory_budget_bytes = 8 * gibibyte;
    }

    auto estimated_dense = dense_bytes(ir);
    if (!estimated_dense)
        return estimated_dense.error();
    plan.estimated_dense_bytes = estimated_dense.value();

    if (ir.layers.empty())
        return Error{ErrorCode::InvalidModel, "memory planner requires at least one layer"};
    const MoeDescriptor& moe = ir.layers.front().ffn.moe;
    if (moe.expert_weight_dtype != DType::MxFp4) {
        plan.selected_mode = ExpertMemoryMode::Eager;
        if (options.expert_memory_mode == ExpertMemoryMode::OnDemand
            || options.expert_cache_bytes != 0) {
            return Error{
                ErrorCode::UnsupportedModel,
                "on-demand expert storage currently requires MXFP4 experts"};
        }
        return plan;
    }

    auto pair_bytes = mxfp4_pair_bytes(ir);
    if (!pair_bytes)
        return pair_bytes.error();
    plan.expert_pair_bytes = pair_bytes.value();
    auto active_bytes = checked_multiply(
        plan.expert_pair_bytes,
        moe.top_k,
        "active experts");
    if (!active_bytes)
        return active_bytes.error();
    plan.minimum_active_expert_bytes = active_bytes.value();
    auto expert_count = checked_multiply(
        ir.layer_count,
        ir.expert_count,
        "expert count");
    if (!expert_count)
        return expert_count.error();
    auto expert_bytes = checked_multiply(
        plan.expert_pair_bytes,
        expert_count.value(),
        "expert weights");
    if (!expert_bytes)
        return expert_bytes.error();
    plan.estimated_expert_bytes = expert_bytes.value();

    const uint64_t safety_reserve = std::max(
        2 * gibibyte,
        physical_memory_bytes == 0 ? 2 * gibibyte : physical_memory_bytes / 8);
    uint64_t eager_capacity = 0;
    if (plan.host_memory_budget_bytes > plan.estimated_dense_bytes
        && plan.host_memory_budget_bytes - plan.estimated_dense_bytes
               > safety_reserve) {
        eager_capacity = plan.host_memory_budget_bytes
                         - plan.estimated_dense_bytes
                         - safety_reserve;
    }

    if (options.expert_cache_bytes != 0) {
        if (options.expert_memory_mode == ExpertMemoryMode::Eager) {
            return Error{
                ErrorCode::InvalidArgument,
                "an explicit expert cache conflicts with eager expert mode"};
        }
        plan.selected_mode = ExpertMemoryMode::OnDemand;
    }
    else if (options.expert_memory_mode == ExpertMemoryMode::Auto) {
        plan.selected_mode
            = plan.estimated_expert_bytes <= eager_capacity
                  ? ExpertMemoryMode::Eager
                  : ExpertMemoryMode::OnDemand;
    }
    else {
        plan.selected_mode = options.expert_memory_mode;
    }

    if (plan.selected_mode == ExpertMemoryMode::Eager)
        return plan;
    if (ir.model_type != "gpt_oss") {
        return Error{
            ErrorCode::UnsupportedModel,
            "on-demand expert storage is implemented for GPT-OSS packages"};
    }
    if (eager_capacity < plan.minimum_active_expert_bytes) {
        return Error{
            ErrorCode::InvalidArgument,
            "host memory budget cannot hold one layer's active Expert set"};
    }

    const uint64_t automatic_target = physical_memory_bytes == 0
                                          ? 2 * gibibyte
                                          : eager_capacity;
    plan.expert_cache_bytes = options.expert_cache_bytes != 0
                                  ? options.expert_cache_bytes
                                  : std::min(automatic_target, eager_capacity);
    if (plan.expert_cache_bytes
        > plan.host_memory_budget_bytes - plan.estimated_dense_bytes) {
        return Error{
            ErrorCode::InvalidArgument,
            "expert cache and dense weights exceed the host memory budget"};
    }
    if (plan.expert_cache_bytes < plan.minimum_active_expert_bytes) {
        return Error{
            ErrorCode::InvalidArgument,
            "expert cache is smaller than one layer's active Expert set"};
    }
    plan.flags |= ModelMemoryFileBackedExperts;
    return plan;
}

} // namespace moe
} // namespace ncnn
