#include "memory_planner.h"

#include "ncnn/moe/runtime.h"

#include "kernels/cpu_mxfp4.h"
#include "kernels/cpu_qnk.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace ncnn {
namespace moe {

static constexpr uint64_t gibibyte = 1024ull * 1024ull * 1024ull;

static Result<uint64_t> checked_add(uint64_t left, uint64_t right, const char* name)
{
    if (right > std::numeric_limits<uint64_t>::max() - left)
        return Error{ErrorCode::InvalidModel, std::string(name) + " byte estimate overflows"};
    return left + right;
}

static Result<uint64_t> checked_multiply(uint64_t left, uint64_t right, const char* name)
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
        return Error{ErrorCode::InvalidModel, std::string(name) + " byte estimate overflows"};
    return left * right;
}

static Result<uint64_t> matrix_storage_bytes(uint64_t rows, uint64_t columns, DType dtype, const char* name)
{
    auto elements = checked_multiply(rows, columns, name);
    if (!elements)
        return elements.error();
    if (dtype == DType::Float8E4M3)
    {
        const uint64_t scale_count = ((rows + 127) / 128) * ((columns + 127) / 128);
        return checked_add(elements.value(), scale_count, name);
    }
    if (is_qnk_dtype(dtype))
    {
        if (rows > std::numeric_limits<uint32_t>::max() || columns > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, std::string(name) + " Qn_K dimensions exceed uint32"};
        const uint64_t bytes = qnk_storage_bytes(dtype, static_cast<size_t>(rows), static_cast<uint32_t>(columns));
        if (bytes == 0)
            return Error{ErrorCode::InvalidModel, std::string(name) + " has invalid Qn_K dimensions"};
        return bytes;
    }
    const uint64_t element_bytes = dtype == DType::Int64 ? 8 : dtype == DType::BFloat16 ? 2
                                                                                        : 4;
    return checked_multiply(elements.value(), element_bytes, name);
}

static Result<void> add_matrix_bytes(uint64_t rows, uint64_t columns, DType dtype, const char* name, uint64_t& total)
{
    auto bytes = matrix_storage_bytes(rows, columns, dtype, name);
    if (!bytes)
        return bytes.error();
    auto added = checked_add(total, bytes.value(), name);
    if (!added)
        return added.error();
    total = added.value();
    return {};
}

static Result<void> add_vector_bytes(uint64_t count, uint64_t element_bytes, const char* name, uint64_t& total)
{
    auto bytes = checked_multiply(count, element_bytes, name);
    if (!bytes)
        return bytes.error();
    auto added = checked_add(total, bytes.value(), name);
    if (!added)
        return added.error();
    total = added.value();
    return {};
}

static Result<uint64_t> latent_fp8_dense_bytes(const MoeIR& ir)
{
    uint64_t total = 0;
    Result<void> status = add_matrix_bytes(ir.vocabulary_size, ir.hidden_size, DType::BFloat16, "DeepSeek embedding", total);
    if (!status)
        return status.error();
    status = add_matrix_bytes(ir.vocabulary_size, ir.hidden_size, DType::BFloat16, "DeepSeek LM head", total);
    if (!status)
        return status.error();
    status = add_vector_bytes(ir.hidden_size, 2, "DeepSeek final norm", total);
    if (!status)
        return status.error();

    const uint64_t multiplier = ir.hyper_connection_multiplier;
    status = add_matrix_bytes(multiplier, multiplier * ir.hidden_size, DType::Float32, "DeepSeek hyper head", total);
    if (!status)
        return status.error();
    status = add_vector_bytes(multiplier + 1, 4, "DeepSeek hyper head parameters", total);
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < ir.layers.size(); ++layer_id)
    {
        const LayerDescriptor& layer = ir.layers[layer_id];
        const AttentionDescriptor& attention = layer.attention;
        const MoeDescriptor& moe = layer.ffn.moe;
        if (attention.kind != AttentionKind::MultiHeadLatent
            || attention.projection_weight_dtype != DType::Float8E4M3
            || attention.output_group_count == 0
            || attention.head_count % attention.output_group_count != 0)
        {
            return Error{ErrorCode::InvalidModel, "invalid DeepSeek latent attention dimensions"};
        }
        const uint64_t mix_count = (2 + multiplier) * multiplier;
        for (uint32_t block = 0; block < 2; ++block)
        {
            status = add_matrix_bytes(mix_count, multiplier * ir.hidden_size, DType::Float32, "DeepSeek hyper function", total);
            if (!status)
                return status.error();
            status = add_vector_bytes(mix_count + 3, 4, "DeepSeek hyper parameters", total);
            if (!status)
                return status.error();
        }
        status = add_vector_bytes(ir.hidden_size * 2ull, 2, "DeepSeek layer norms", total);
        if (!status)
            return status.error();

        status = add_matrix_bytes(attention.query_lora_rank, ir.hidden_size, attention.projection_weight_dtype, "DeepSeek query A", total);
        if (!status)
            return status.error();
        status = add_matrix_bytes(static_cast<uint64_t>(attention.head_count) * attention.head_dimension, attention.query_lora_rank, attention.projection_weight_dtype, "DeepSeek query B", total);
        if (!status)
            return status.error();
        status = add_matrix_bytes(attention.head_dimension, ir.hidden_size, attention.projection_weight_dtype, "DeepSeek latent KV", total);
        if (!status)
            return status.error();
        const uint64_t group_input = static_cast<uint64_t>(attention.head_count) * attention.head_dimension / attention.output_group_count;
        status = add_matrix_bytes(static_cast<uint64_t>(attention.output_group_count) * attention.output_lora_rank, group_input, attention.projection_weight_dtype, "DeepSeek output A", total);
        if (!status)
            return status.error();
        status = add_matrix_bytes(ir.hidden_size, static_cast<uint64_t>(attention.output_group_count) * attention.output_lora_rank, attention.projection_weight_dtype, "DeepSeek output B", total);
        if (!status)
            return status.error();
        status = add_vector_bytes(attention.query_lora_rank + attention.head_dimension, 2, "DeepSeek attention norms", total);
        if (!status)
            return status.error();
        status = add_vector_bytes(attention.head_count, 4, "DeepSeek attention sinks", total);
        if (!status)
            return status.error();

        if (attention.compression_ratio != 0)
        {
            const uint64_t compressor_multiplier = attention.compression_ratio == 4 ? 2 : 1;
            status = add_matrix_bytes(compressor_multiplier * attention.head_dimension, ir.hidden_size, DType::BFloat16, "DeepSeek compressor KV", total);
            if (!status)
                return status.error();
            status = add_matrix_bytes(compressor_multiplier * attention.head_dimension, ir.hidden_size, DType::BFloat16, "DeepSeek compressor gate", total);
            if (!status)
                return status.error();
            status = add_matrix_bytes(attention.compression_ratio, compressor_multiplier * attention.head_dimension, DType::Float32, "DeepSeek compressor position", total);
            if (!status)
                return status.error();
            status = add_vector_bytes(attention.head_dimension, 2, "DeepSeek compressor norm", total);
            if (!status)
                return status.error();
            if (attention.compression_ratio == 4)
            {
                status = add_matrix_bytes(2ull * attention.index_head_dimension, ir.hidden_size, DType::BFloat16, "DeepSeek index compressor KV", total);
                if (!status)
                    return status.error();
                status = add_matrix_bytes(2ull * attention.index_head_dimension, ir.hidden_size, DType::BFloat16, "DeepSeek index compressor gate", total);
                if (!status)
                    return status.error();
                status = add_matrix_bytes(attention.compression_ratio, 2ull * attention.index_head_dimension, DType::Float32, "DeepSeek index compressor position", total);
                if (!status)
                    return status.error();
                status = add_matrix_bytes(static_cast<uint64_t>(attention.index_head_count) * attention.index_head_dimension, attention.query_lora_rank, attention.projection_weight_dtype, "DeepSeek index query", total);
                if (!status)
                    return status.error();
                status = add_matrix_bytes(attention.index_head_count, ir.hidden_size, DType::BFloat16, "DeepSeek index weights", total);
                if (!status)
                    return status.error();
                status = add_vector_bytes(attention.index_head_dimension, 2, "DeepSeek index norm", total);
                if (!status)
                    return status.error();
            }
        }

        status = add_matrix_bytes(moe.expert_count, ir.hidden_size, DType::BFloat16, "DeepSeek router", total);
        if (!status)
            return status.error();
        if (layer_id < ir.hash_routing_layer_count)
            status = add_matrix_bytes(ir.vocabulary_size, moe.top_k, DType::Int64, "DeepSeek hash router", total);
        else
            status = add_vector_bytes(moe.expert_count, 4, "DeepSeek router bias", total);
        if (!status)
            return status.error();
        status = add_matrix_bytes(moe.intermediate_size, ir.hidden_size, moe.shared_expert_weight_dtype, "DeepSeek shared Expert input", total);
        if (!status)
            return status.error();
        status = add_matrix_bytes(moe.intermediate_size, ir.hidden_size, moe.shared_expert_weight_dtype, "DeepSeek shared Expert input", total);
        if (!status)
            return status.error();
        status = add_matrix_bytes(ir.hidden_size, moe.intermediate_size, moe.shared_expert_weight_dtype, "DeepSeek shared Expert output", total);
        if (!status)
            return status.error();
    }
    return total;
}

static Result<uint64_t> latent_vulkan_releasable_dense_bytes(const MoeIR& ir)
{
    if (ir.layers.empty()
        || ir.hyper_connection_multiplier <= 1
        || ir.layers.front().attention.kind != AttentionKind::MultiHeadLatent
        || ir.layers.front().attention.projection_weight_dtype != DType::Float8E4M3)
    {
        return uint64_t{0};
    }

    uint64_t total = 0;
    Result<void> status = add_matrix_bytes(
        ir.vocabulary_size,
        ir.hidden_size,
        DType::BFloat16,
        "Vulkan LM head",
        total);
    if (!status)
        return status.error();

    for (const LayerDescriptor& layer : ir.layers)
    {
        const AttentionDescriptor& attention = layer.attention;
        if (attention.kind != AttentionKind::MultiHeadLatent
            || attention.projection_weight_dtype != DType::Float8E4M3
            || attention.output_group_count == 0
            || attention.head_count % attention.output_group_count != 0)
        {
            return uint64_t{0};
        }
        const uint64_t group_input = static_cast<uint64_t>(attention.head_count) * attention.head_dimension / attention.output_group_count;
        const std::pair<uint64_t, uint64_t> matrices[] = {
            {attention.query_lora_rank, ir.hidden_size},
            {static_cast<uint64_t>(attention.head_count) * attention.head_dimension, attention.query_lora_rank},
            {attention.head_dimension, ir.hidden_size},
            {static_cast<uint64_t>(attention.output_group_count) * attention.output_lora_rank, group_input},
            {ir.hidden_size, static_cast<uint64_t>(attention.output_group_count) * attention.output_lora_rank},
        };
        for (const auto& matrix : matrices)
        {
            status = add_matrix_bytes(
                matrix.first,
                matrix.second,
                DType::Float8E4M3,
                "Vulkan latent projection",
                total);
            if (!status)
                return status.error();
        }
        if (attention.compression_ratio == 4)
        {
            status = add_matrix_bytes(
                static_cast<uint64_t>(attention.index_head_count) * attention.index_head_dimension,
                attention.query_lora_rank,
                DType::Float8E4M3,
                "Vulkan index query",
                total);
            if (!status)
                return status.error();
        }

        if (layer.ffn.moe.shared_expert_count != 0)
        {
            status = add_matrix_bytes(
                layer.ffn.moe.intermediate_size,
                ir.hidden_size,
                layer.ffn.moe.shared_expert_weight_dtype,
                "Vulkan shared Expert input",
                total);
            if (!status)
                return status.error();
            status = add_matrix_bytes(
                layer.ffn.moe.intermediate_size,
                ir.hidden_size,
                layer.ffn.moe.shared_expert_weight_dtype,
                "Vulkan shared Expert input",
                total);
            if (!status)
                return status.error();
            status = add_matrix_bytes(
                ir.hidden_size,
                layer.ffn.moe.intermediate_size,
                layer.ffn.moe.shared_expert_weight_dtype,
                "Vulkan shared Expert output",
                total);
            if (!status)
                return status.error();
        }
    }
    return total;
}

static Result<uint64_t> dense_bytes(const MoeIR& ir)
{
    if (!ir.layers.empty()
        && ir.hyper_connection_multiplier > 1
        && ir.layers.front().attention.kind == AttentionKind::MultiHeadLatent
        && ir.layers.front().attention.projection_weight_dtype == DType::Float8E4M3)
        return latent_fp8_dense_bytes(ir);
    const uint64_t element_bytes = ir.activation_dtype == DType::BFloat16 ? 2 : 4;
    auto embedding_elements = checked_multiply(ir.vocabulary_size, ir.hidden_size, "embedding");
    if (!embedding_elements)
        return embedding_elements.error();
    auto embedding_bytes = checked_multiply(embedding_elements.value(), element_bytes * 2, "embedding and LM head");
    if (!embedding_bytes)
        return embedding_bytes.error();
    uint64_t total = embedding_bytes.value();

    for (const LayerDescriptor& layer : ir.layers)
    {
        uint64_t layer_elements = ir.hidden_size;

        if (has_flag(layer.flags, LayerDescriptorAttention))
        {
            auto with_attention_norm = checked_add(layer_elements, ir.hidden_size, "layer norms");
            if (!with_attention_norm)
                return with_attention_norm.error();
            layer_elements = with_attention_norm.value();
            const AttentionDescriptor& attention = layer.attention;
            const uint64_t query_size = static_cast<uint64_t>(attention.head_count) * attention.head_dimension;
            const uint64_t key_value_size = static_cast<uint64_t>(attention.kv_head_count) * attention.head_dimension;
            Result<uint64_t> query = uint64_t{0};
            Result<uint64_t> key_value = uint64_t{0};
            Result<uint64_t> output = uint64_t{0};
            if (attention.kind == AttentionKind::Standard)
            {
                query = checked_multiply(query_size, ir.hidden_size, "query projection");
                key_value = checked_multiply(key_value_size, ir.hidden_size * 2ull, "key/value projections");
                output = checked_multiply(ir.hidden_size, query_size, "attention output projection");
            }
            else if (attention.kind == AttentionKind::MultiHeadLatent)
            {
                const uint64_t qk_size = static_cast<uint64_t>(attention.head_count) * (attention.qk_nope_head_dimension + attention.qk_rope_head_dimension);
                const uint64_t kv_a_size = attention.kv_lora_rank + attention.qk_rope_head_dimension;
                const uint64_t kv_b_size = static_cast<uint64_t>(attention.head_count) * (attention.qk_nope_head_dimension + attention.value_head_dimension);
                const uint64_t query_a_elements = attention.query_lora_rank == 0
                                                      ? 0
                                                      : static_cast<uint64_t>(ir.hidden_size) * attention.query_lora_rank + attention.query_lora_rank;
                const uint64_t query_b_elements = attention.query_lora_rank == 0
                                                      ? static_cast<uint64_t>(ir.hidden_size) * qk_size
                                                      : static_cast<uint64_t>(attention.query_lora_rank) * qk_size;
                query = checked_add(query_a_elements, query_b_elements, "MLA query projections");
                key_value = checked_add(static_cast<uint64_t>(ir.hidden_size) * kv_a_size + attention.kv_lora_rank, static_cast<uint64_t>(attention.kv_lora_rank) * kv_b_size, "MLA key/value projections");
                output = checked_multiply(ir.hidden_size, static_cast<uint64_t>(attention.head_count) * attention.value_head_dimension, "MLA output projection");
            }
            else
            {
                const uint64_t linear_key_size = static_cast<uint64_t>(attention.kv_head_count) * attention.head_dimension;
                const uint64_t linear_value_size = static_cast<uint64_t>(attention.head_count) * attention.value_head_dimension;
                const uint64_t convolution_size = linear_key_size * 2 + linear_value_size;
                query = checked_multiply(convolution_size + linear_value_size + attention.head_count * 2ull, ir.hidden_size, "gated DeltaNet projections");
                key_value = checked_multiply(convolution_size, attention.convolution_kernel_size, "gated DeltaNet convolution");
                if (key_value)
                {
                    key_value = checked_add(
                        key_value.value(),
                        attention.head_count * 2ull + attention.value_head_dimension,
                        "gated DeltaNet parameters");
                }
                output = checked_multiply(ir.hidden_size, linear_value_size, "gated DeltaNet output projection");
            }
            if (!query)
                return query.error();
            if (!key_value)
                return key_value.error();
            if (!output)
                return output.error();
            auto projections = checked_add(query.value(), key_value.value(), "attention projections");
            if (!projections)
                return projections.error();
            projections = checked_add(projections.value(), output.value(), "attention projections");
            if (!projections)
                return projections.error();
            auto with_projections = checked_add(layer_elements, projections.value(), "layer dense weights");
            if (!with_projections)
                return with_projections.error();
            layer_elements = with_projections.value();
            if (has_flag(layer.attention.flags, AttentionDescriptorBias))
            {
                const uint64_t bias_elements = attention.kind == AttentionKind::Standard
                                                   ? query_size + key_value_size * 2 + ir.hidden_size
                                                   : static_cast<uint64_t>(attention.query_lora_rank) + attention.kv_lora_rank + attention.qk_rope_head_dimension + ir.hidden_size;
                auto with_bias = checked_add(layer_elements, bias_elements, "attention biases");
                if (!with_bias)
                    return with_bias.error();
                layer_elements = with_bias.value();
            }
            if (has_flag(layer.attention.flags, AttentionDescriptorSinks))
            {
                auto with_sinks = checked_add(layer_elements, layer.attention.head_count, "attention sinks");
                if (!with_sinks)
                    return with_sinks.error();
                layer_elements = with_sinks.value();
            }
            if (has_flag(layer.attention.flags, AttentionDescriptorQueryKeyNorm))
            {
                auto with_qk_norm = checked_add(layer_elements, layer.attention.head_dimension * 2ull, "query/key norms");
                if (!with_qk_norm)
                    return with_qk_norm.error();
                layer_elements = with_qk_norm.value();
            }
            if (has_flag(layer.attention.flags, AttentionDescriptorOutputGate))
            {
                auto output_gate = checked_multiply(query_size, ir.hidden_size, "attention output gate");
                if (!output_gate)
                    return output_gate.error();
                auto with_output_gate = checked_add(layer_elements, output_gate.value(), "attention output gate");
                if (!with_output_gate)
                    return with_output_gate.error();
                layer_elements = with_output_gate.value();
            }
            if (has_flag(layer.attention.flags, AttentionDescriptorQsa))
            {
                const uint64_t index_columns =
                    (static_cast<uint64_t>(attention.index_head_count) + 1)
                    * attention.index_head_dimension;
                auto index_projection = checked_multiply(
                    index_columns, ir.hidden_size, "QSA index projection");
                if (!index_projection)
                    return index_projection.error();
                auto index_weights = checked_add(
                    index_projection.value(),
                    static_cast<uint64_t>(attention.index_head_dimension) * 2,
                    "QSA index weights");
                if (!index_weights)
                    return index_weights.error();
                auto with_index = checked_add(
                    layer_elements, index_weights.value(),
                    "layer dense weights");
                if (!with_index)
                    return with_index.error();
                layer_elements = with_index.value();
            }
        }

        uint64_t layer_auxiliary_bytes = 0;
        if (layer.ple.enabled())
        {
            if (layer.ple.ngram_size < 2 || layer.ple.heads_per_ngram == 0)
                return Error{ErrorCode::InvalidModel, "invalid PLE dense-memory configuration"};
            const uint64_t head_count =
                static_cast<uint64_t>(layer.ple.ngram_size - 1)
                * layer.ple.heads_per_ngram;
            if (head_count == 0
                || layer.ple.embedding_dimension != ir.hidden_size
                || layer.ple.embedding_dimension % head_count != 0)
            {
                return Error{ErrorCode::InvalidModel, "invalid PLE dense-memory configuration"};
            }
            const uint64_t expanded_size =
                static_cast<uint64_t>(ir.hyper_connection_multiplier)
                * ir.hidden_size;
            auto key_projection = checked_multiply(
                expanded_size, ir.hidden_size, "PLE key projection");
            if (!key_projection)
                return key_projection.error();
            auto value_projection = checked_multiply(
                ir.hidden_size, ir.hidden_size, "PLE value projection");
            if (!value_projection)
                return value_projection.error();
            auto ple_elements = checked_add(
                key_projection.value(), value_projection.value(),
                "PLE projections");
            if (!ple_elements)
                return ple_elements.error();
            auto norm_and_convolution = checked_multiply(
                expanded_size,
                3ull + layer.ple.convolution_kernel_size,
                "PLE norms and convolution");
            if (!norm_and_convolution)
                return norm_and_convolution.error();
            ple_elements = checked_add(
                ple_elements.value(), norm_and_convolution.value(),
                "PLE dense weights");
            if (!ple_elements)
                return ple_elements.error();
            auto with_ple = checked_add(
                layer_elements, ple_elements.value(),
                "layer dense weights");
            if (!with_ple)
                return with_ple.error();
            layer_elements = with_ple.value();
            auto head_metadata_count = checked_multiply(
                head_count, 2, "PLE metadata");
            if (!head_metadata_count)
                return head_metadata_count.error();
            auto metadata_count = checked_add(
                layer.ple.ngram_size, head_metadata_count.value(),
                "PLE metadata");
            if (!metadata_count)
                return metadata_count.error();
            auto metadata_bytes = checked_multiply(
                metadata_count.value(), sizeof(int64_t),
                "PLE metadata");
            if (!metadata_bytes)
                return metadata_bytes.error();
            layer_auxiliary_bytes = metadata_bytes.value();
        }

        if (ir.hyper_connection_kind == HyperConnectionKind::GatedResidual)
        {
            const uint64_t expanded_size =
                static_cast<uint64_t>(ir.hyper_connection_multiplier)
                * ir.hidden_size;
            auto projections = checked_multiply(
                expanded_size, ir.hyper_connection_low_rank * 2ull,
                "gated-residual projections");
            if (!projections)
                return projections.error();
            auto block_elements = checked_add(
                expanded_size, projections.value(),
                "gated-residual block");
            if (!block_elements)
                return block_elements.error();
            auto injection = checked_multiply(
                ir.hyper_connection_multiplier, expanded_size,
                "gated-residual injection");
            if (!injection)
                return injection.error();
            block_elements = checked_add(
                block_elements.value(), injection.value(),
                "gated-residual block");
            if (!block_elements)
                return block_elements.error();
            auto layer_gated_residual = checked_multiply(
                block_elements.value(), 2, "gated-residual layer");
            if (!layer_gated_residual)
                return layer_gated_residual.error();
            auto with_gated_residual = checked_add(
                layer_elements, layer_gated_residual.value(),
                "layer dense weights");
            if (!with_gated_residual)
                return with_gated_residual.error();
            layer_elements = with_gated_residual.value();
        }

        if (has_flag(layer.flags, LayerDescriptorMoe))
        {
            auto router = checked_multiply(layer.ffn.moe.expert_count, ir.hidden_size, "router");
            if (!router)
                return router.error();
            auto with_router = checked_add(layer_elements, router.value(), "layer dense weights");
            if (!with_router)
                return with_router.error();
            layer_elements = with_router.value();
            if (has_flag(layer.ffn.moe.flags, MoeDescriptorRouterBias))
            {
                auto with_router_bias = checked_add(layer_elements, layer.ffn.moe.expert_count, "router bias");
                if (!with_router_bias)
                    return with_router_bias.error();
                layer_elements = with_router_bias.value();
            }
            if (has_flag(layer.ffn.moe.flags, MoeDescriptorProjectionBias))
            {
                const uint64_t per_expert_bias = static_cast<uint64_t>(layer.ffn.moe.intermediate_size) * 2 + ir.hidden_size;
                auto expert_biases = checked_multiply(layer.ffn.moe.expert_count, per_expert_bias, "expert biases");
                if (!expert_biases)
                    return expert_biases.error();
                auto with_expert_biases = checked_add(layer_elements, expert_biases.value(), "layer dense weights");
                if (!with_expert_biases)
                    return with_expert_biases.error();
                layer_elements = with_expert_biases.value();
            }
            if (layer.ffn.moe.shared_expert_count != 0)
            {
                auto shared_elements = checked_multiply(static_cast<uint64_t>(ir.hidden_size) * layer.ffn.moe.intermediate_size * 3, layer.ffn.moe.shared_expert_count, "shared Expert weights");
                if (!shared_elements)
                    return shared_elements.error();
                auto with_shared = checked_add(layer_elements, shared_elements.value(), "layer dense weights");
                if (!with_shared)
                    return with_shared.error();
                layer_elements = with_shared.value();
                if (has_flag(layer.ffn.moe.flags, MoeDescriptorSharedExpertGate))
                {
                    auto with_shared_gate = checked_add(layer_elements, ir.hidden_size, "shared Expert gate");
                    if (!with_shared_gate)
                        return with_shared_gate.error();
                    layer_elements = with_shared_gate.value();
                }
            }
        }
        else if (has_flag(layer.flags, LayerDescriptorDenseFfn))
        {
            auto dense_ffn_elements = checked_multiply(static_cast<uint64_t>(ir.hidden_size) * layer.ffn.dense_intermediate_size, 3, "dense FFN weights");
            if (!dense_ffn_elements)
                return dense_ffn_elements.error();
            auto with_dense_ffn = checked_add(layer_elements, dense_ffn_elements.value(), "layer dense weights");
            if (!with_dense_ffn)
                return with_dense_ffn.error();
            layer_elements = with_dense_ffn.value();
        }

        auto layer_bytes = checked_multiply(layer_elements, element_bytes, "layer dense weights");
        if (!layer_bytes)
            return layer_bytes.error();
        auto with_layer = checked_add(total, layer_bytes.value(), "dense weights");
        if (!with_layer)
            return with_layer.error();
        with_layer = checked_add(
            with_layer.value(), layer_auxiliary_bytes,
            "dense weights");
        if (!with_layer)
            return with_layer.error();
        total = with_layer.value();
    }

    if (ir.speculative_kind == SpeculativeModelKind::Mtp)
    {
        if (ir.speculative_layer_count != 1
            || ir.layers.empty()
            || ir.layers.back().attention.kind != AttentionKind::Standard)
        {
            return Error{ErrorCode::InvalidModel, "invalid MTP dense-memory configuration"};
        }
        const LayerDescriptor& layer = ir.layers.back();
        const AttentionDescriptor& attention = layer.attention;
        const MoeDescriptor& moe = layer.ffn.moe;
        const uint64_t query_size = static_cast<uint64_t>(attention.head_count) * attention.head_dimension;
        const uint64_t key_value_size = static_cast<uint64_t>(attention.kv_head_count) * attention.head_dimension;
        uint64_t mtp_elements = 0;
        auto add_elements = [&mtp_elements](uint64_t elements, const char* name) -> Result<void> {
            auto added = checked_add(mtp_elements, elements, name);
            if (!added)
                return added.error();
            mtp_elements = added.value();
            return {};
        };
        auto matrix_elements = [](uint64_t rows, uint64_t columns, const char* name) {
            return checked_multiply(rows, columns, name);
        };

        Result<void> status = add_elements(ir.hidden_size * 5ull, "MTP norms");
        if (!status)
            return status.error();
        const std::pair<uint64_t, uint64_t> matrices[] = {
            {ir.hidden_size, ir.hidden_size * 2ull},
            {query_size * 2ull, ir.hidden_size},
            {key_value_size * 2ull, ir.hidden_size},
            {ir.hidden_size, query_size},
            {moe.expert_count, ir.hidden_size},
        };
        for (const auto& matrix : matrices)
        {
            auto elements = matrix_elements(matrix.first, matrix.second, "MTP matrix");
            if (!elements)
                return elements.error();
            status = add_elements(elements.value(), "MTP matrix");
            if (!status)
                return status.error();
        }
        status = add_elements(attention.head_dimension * 2ull, "MTP query/key norms");
        if (!status)
            return status.error();
        if (moe.shared_expert_count != 0)
        {
            auto shared = matrix_elements(
                static_cast<uint64_t>(ir.hidden_size) * moe.intermediate_size * 3ull,
                moe.shared_expert_count,
                "MTP shared Expert");
            if (!shared)
                return shared.error();
            status = add_elements(shared.value(), "MTP shared Expert");
            if (!status)
                return status.error();
            if (has_flag(moe.flags, MoeDescriptorSharedExpertGate))
            {
                status = add_elements(ir.hidden_size, "MTP shared Expert gate");
                if (!status)
                    return status.error();
            }
        }
        auto mtp_bytes = checked_multiply(mtp_elements, element_bytes, "MTP dense weights");
        if (!mtp_bytes)
            return mtp_bytes.error();
        auto with_mtp = checked_add(total, mtp_bytes.value(), "dense weights");
        if (!with_mtp)
            return with_mtp.error();
        total = with_mtp.value();
    }

    if (ir.hyper_connection_kind == HyperConnectionKind::GatedResidual)
    {
        const uint64_t expanded_size =
            static_cast<uint64_t>(ir.hyper_connection_multiplier)
            * ir.hidden_size;
        auto head_projections = checked_multiply(
            expanded_size, ir.hyper_connection_low_rank * 2ull,
            "gated-residual head projections");
        if (!head_projections)
            return head_projections.error();
        auto head_elements = checked_add(
            expanded_size, head_projections.value(),
            "gated-residual head");
        if (!head_elements)
            return head_elements.error();
        auto head_bytes = checked_multiply(
            head_elements.value(), element_bytes,
            "gated-residual head");
        if (!head_bytes)
            return head_bytes.error();
        auto with_head = checked_add(total, head_bytes.value(), "dense weights");
        if (!with_head)
            return with_head.error();
        total = with_head.value();
    }

    const uint64_t final_norm_elements = ir.final_norm == NormType::None
                                             ? 0
                                             : ir.hidden_size;
    auto final_norm = checked_multiply(
        final_norm_elements, element_bytes, "final norm");
    if (!final_norm)
        return final_norm.error();
    return checked_add(total, final_norm.value(), "dense weights");
}

static Result<uint64_t> mxfp4_pair_bytes(const MoeIR& ir)
{
    auto elements = checked_multiply(ir.hidden_size, ir.intermediate_size, "expert pair");
    if (!elements)
        return elements.error();
    elements = checked_multiply(elements.value(), 3, "expert pair");
    if (!elements)
        return elements.error();
    auto encoded = checked_multiply(elements.value(), 17, "MXFP4 expert pair");
    if (!encoded)
        return encoded.error();
    return encoded.value() / 32;
}

static Result<uint64_t> expert_pair_bytes(const MoeIR& ir, DType dtype)
{
    if (dtype == DType::MxFp4)
        return mxfp4_pair_bytes(ir);
    auto gate_up = matrix_storage_bytes(
        static_cast<uint64_t>(ir.intermediate_size) * 2,
        ir.hidden_size,
        dtype,
        "expert gate/up");
    if (!gate_up)
        return gate_up.error();
    auto down = matrix_storage_bytes(
        ir.hidden_size,
        ir.intermediate_size,
        dtype,
        "expert down");
    if (!down)
        return down.error();
    return checked_add(gate_up.value(), down.value(), "expert pair");
}

static Result<uint64_t> packed_matrix_storage_bytes(
    uint64_t rows,
    uint64_t columns,
    DType dtype,
    const char* name)
{
    if (dtype == DType::MxFp4)
    {
        if (rows < 4)
            return uint64_t{0};
        if (rows > std::numeric_limits<size_t>::max()
            || columns == 0
            || columns > std::numeric_limits<uint32_t>::max()
            || columns % 32 != 0)
        {
            return Error{ErrorCode::InvalidModel, std::string(name) + " has invalid MXFP4 dimensions"};
        }
        const uint64_t bytes = mxfp4_q8_packed_storage_bytes(
            static_cast<size_t>(rows),
            static_cast<uint32_t>(columns / 32));
        if (bytes == 0)
            return Error{ErrorCode::InvalidModel, std::string(name) + " packed byte estimate overflows"};
        return bytes;
    }
    if (is_qnk_dtype(dtype))
    {
        if (rows > std::numeric_limits<size_t>::max()
            || columns > std::numeric_limits<uint32_t>::max())
        {
            return Error{ErrorCode::InvalidModel, std::string(name) + " has invalid Qn_K dimensions"};
        }
        const uint64_t bytes = qnk_packed_storage_bytes(
            dtype,
            static_cast<size_t>(rows),
            static_cast<uint32_t>(columns));
        if (bytes == 0)
            return Error{ErrorCode::InvalidModel, std::string(name) + " has invalid Qn_K dimensions"};
        return bytes;
    }
    return uint64_t{0};
}

static Result<uint64_t> packed_expert_pair_bytes(const MoeIR& ir, DType dtype)
{
    auto gate_up = packed_matrix_storage_bytes(
        static_cast<uint64_t>(ir.intermediate_size) * 2,
        ir.hidden_size,
        dtype,
        "packed expert gate/up");
    if (!gate_up)
        return gate_up.error();
    auto down = packed_matrix_storage_bytes(
        ir.hidden_size,
        ir.intermediate_size,
        dtype,
        "packed expert down");
    if (!down)
        return down.error();
    return checked_add(gate_up.value(), down.value(), "packed expert pair");
}

Result<ModelMemoryPlan> plan_model_memory(const MoeIR& ir, const RuntimeConfig& config, uint64_t physical_memory_bytes,
                                          bool release_vulkan_dense_host_storage,
                                          uint64_t available_memory_bytes,
                                          bool reserve_cpu_packed_weights)
{
    ModelMemoryPlan plan;
    bool budget_limited_by_available_memory = false;
    plan.requested_mode = config.expert_memory_mode;
    plan.physical_memory_bytes = physical_memory_bytes;
    plan.available_memory_bytes = available_memory_bytes;
    if (config.host_memory_budget_bytes != 0)
    {
        if (physical_memory_bytes != 0 && config.host_memory_budget_bytes > physical_memory_bytes)
        {
            return Error{ErrorCode::InvalidArgument, "host memory budget exceeds detected physical memory"};
        }
        plan.host_memory_budget_bytes = config.host_memory_budget_bytes;
    }
    else if (physical_memory_bytes != 0)
    {
        plan.host_memory_budget_bytes = physical_memory_bytes / 4 * 3;
        if (available_memory_bytes != 0)
        {
            const uint64_t system_reserve = 2 * gibibyte;
            const uint64_t available_budget =
                available_memory_bytes > system_reserve
                    ? available_memory_bytes - system_reserve
                    : available_memory_bytes / 2;
            if (available_budget < plan.host_memory_budget_bytes)
            {
                plan.host_memory_budget_bytes = available_budget;
                budget_limited_by_available_memory = true;
            }
        }
    }
    else
    {
        plan.host_memory_budget_bytes = 8 * gibibyte;
    }

    auto estimated_dense = dense_bytes(ir);
    if (!estimated_dense)
        return estimated_dense.error();
    // PLE embedding shards remain file-backed and are paged on demand.  The
    // dense estimate covers only their resident projections and state.
    plan.estimated_dense_bytes = estimated_dense.value();

    if (ir.layers.empty())
        return Error{ErrorCode::InvalidModel, "memory planner requires at least one layer"};
    const LayerDescriptor* first_moe_layer = nullptr;
    uint64_t moe_layer_count = 0;
    for (const LayerDescriptor& layer : ir.layers)
    {
        if (!has_flag(layer.flags, LayerDescriptorMoe))
            continue;
        if (!first_moe_layer)
            first_moe_layer = &layer;
        ++moe_layer_count;
    }
    moe_layer_count += ir.speculative_layer_count;
    if (!first_moe_layer)
        return Error{ErrorCode::InvalidModel, "memory planner requires at least one MoE layer"};
    const MoeDescriptor& moe = first_moe_layer->ffn.moe;
    auto pair_bytes = expert_pair_bytes(ir, moe.expert_weight_dtype);
    if (!pair_bytes)
        return pair_bytes.error();
    plan.expert_pair_bytes = pair_bytes.value();
    plan.expert_pair_resident_bytes = plan.expert_pair_bytes;
    if (reserve_cpu_packed_weights)
    {
        auto packed_pair_bytes = packed_expert_pair_bytes(ir, moe.expert_weight_dtype);
        if (!packed_pair_bytes)
            return packed_pair_bytes.error();
        auto resident_pair_bytes = checked_add(
            plan.expert_pair_bytes,
            packed_pair_bytes.value(),
            "resident expert pair");
        if (!resident_pair_bytes)
            return resident_pair_bytes.error();
        plan.expert_pair_resident_bytes = resident_pair_bytes.value();
    }
    auto active_bytes = checked_multiply(plan.expert_pair_resident_bytes, moe.top_k, "active experts");
    if (!active_bytes)
        return active_bytes.error();
    plan.minimum_active_expert_bytes = active_bytes.value();
    auto expert_count = checked_multiply(moe_layer_count, ir.expert_count, "expert count");
    if (!expert_count)
        return expert_count.error();
    auto expert_bytes = checked_multiply(plan.expert_pair_bytes, expert_count.value(), "expert weights");
    if (!expert_bytes)
        return expert_bytes.error();
    plan.estimated_expert_bytes = expert_bytes.value();
    auto packed_expert_bytes = checked_multiply(
        plan.expert_pair_resident_bytes - plan.expert_pair_bytes,
        expert_count.value(),
        "packed expert weights");
    if (!packed_expert_bytes)
        return packed_expert_bytes.error();
    plan.estimated_cpu_packed_expert_bytes = packed_expert_bytes.value();
    auto resident_expert_bytes = checked_add(
        plan.estimated_expert_bytes,
        plan.estimated_cpu_packed_expert_bytes,
        "resident expert weights");
    if (!resident_expert_bytes)
        return resident_expert_bytes.error();
    plan.estimated_expert_resident_bytes = resident_expert_bytes.value();

    const uint64_t safety_reserve = budget_limited_by_available_memory
                                        ? 0
                                        : std::max(2 * gibibyte, physical_memory_bytes == 0 ? 2 * gibibyte : physical_memory_bytes / 8);
    uint64_t eager_capacity = 0;
    if (plan.host_memory_budget_bytes > plan.estimated_dense_bytes && plan.host_memory_budget_bytes - plan.estimated_dense_bytes > safety_reserve)
    {
        eager_capacity = plan.host_memory_budget_bytes - plan.estimated_dense_bytes - safety_reserve;
    }
    const auto supports_file_backed_experts = [](const MoeDescriptor& descriptor) {
        return descriptor.expert_weight_dtype == DType::MxFp4
               || (descriptor.expert_weight_dtype == DType::BFloat16
                   && has_flag(descriptor.flags, MoeDescriptorFileBackedExperts));
    };
    if (!supports_file_backed_experts(moe))
    {
        plan.selected_mode = ExpertMemoryMode::Eager;
        if (config.expert_memory_mode == ExpertMemoryMode::OnDemand || config.expert_cache_bytes != 0)
        {
            return Error{ErrorCode::UnsupportedModel, "on-demand expert storage requires an explicitly file-backed Expert encoding"};
        }
        if (reserve_cpu_packed_weights
            && plan.estimated_expert_resident_bytes > eager_capacity)
        {
            return Error{ErrorCode::InvalidArgument, "host memory budget cannot hold the requested CPU packed Expert weights"};
        }
        return plan;
    }

    if (config.expert_cache_bytes != 0)
    {
        if (config.expert_memory_mode == ExpertMemoryMode::Eager)
        {
            return Error{ErrorCode::InvalidArgument, "an explicit expert cache conflicts with eager expert mode"};
        }
        plan.selected_mode = ExpertMemoryMode::OnDemand;
    }
    else if (config.expert_memory_mode == ExpertMemoryMode::Auto)
    {
        plan.selected_mode = plan.estimated_expert_resident_bytes <= eager_capacity ? ExpertMemoryMode::Eager : ExpertMemoryMode::OnDemand;
    }
    else
    {
        plan.selected_mode = config.expert_memory_mode;
    }

    if (plan.selected_mode == ExpertMemoryMode::Eager)
    {
        if (reserve_cpu_packed_weights
            && plan.estimated_expert_resident_bytes > eager_capacity)
        {
            return Error{ErrorCode::InvalidArgument, "host memory budget cannot hold the requested CPU packed Expert weights"};
        }
        return plan;
    }
    bool file_backed_expert_encoding = true;
    for (const LayerDescriptor& layer : ir.layers)
    {
        if (has_flag(layer.flags, LayerDescriptorMoe)
            && !supports_file_backed_experts(layer.ffn.moe))
        {
            file_backed_expert_encoding = false;
            break;
        }
    }
    if (!file_backed_expert_encoding)
    {
        return Error{ErrorCode::UnsupportedModel, "on-demand mode requires a file-backed Expert encoding"};
    }

    if (release_vulkan_dense_host_storage)
    {
        auto releasable = latent_vulkan_releasable_dense_bytes(ir);
        if (!releasable)
            return releasable.error();
        plan.estimated_dense_bytes = releasable.value() >= plan.estimated_dense_bytes
                                         ? 0
                                         : plan.estimated_dense_bytes - releasable.value();
        eager_capacity = 0;
        if (plan.host_memory_budget_bytes > plan.estimated_dense_bytes
            && plan.host_memory_budget_bytes - plan.estimated_dense_bytes > safety_reserve)
        {
            eager_capacity = plan.host_memory_budget_bytes - plan.estimated_dense_bytes - safety_reserve;
        }
    }
    if (eager_capacity < plan.minimum_active_expert_bytes)
    {
        return Error{ErrorCode::InvalidArgument, "host memory budget cannot hold one layer's active Expert set"};
    }

    const uint64_t automatic_target = physical_memory_bytes == 0 ? 2 * gibibyte : eager_capacity;
    plan.expert_cache_bytes = config.expert_cache_bytes != 0 ? config.expert_cache_bytes : std::min(automatic_target, eager_capacity);
    if (plan.expert_cache_bytes > plan.host_memory_budget_bytes - plan.estimated_dense_bytes)
    {
        return Error{ErrorCode::InvalidArgument, "expert cache and dense weights exceed the host memory budget"};
    }
    if (plan.expert_cache_bytes < plan.minimum_active_expert_bytes)
    {
        return Error{ErrorCode::InvalidArgument, "expert cache is smaller than one layer's active Expert set"};
    }
    plan.flags |= ModelMemoryFileBackedExperts;
    return plan;
}

} // namespace moe
} // namespace ncnn
